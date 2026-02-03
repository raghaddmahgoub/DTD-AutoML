
from typing import TypedDict, Annotated, Literal, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import pandas as pd
import numpy as np
from pathlib import Path
import os
import time
from dotenv import load_dotenv

from src.utils.data_analyzer import DataAnalyzer
from src.utils.logger import Logger
# Load environment variables
load_dotenv()

logger = Logger()


class AgentState(TypedDict):
    """State of the AutoML agent throughout the workflow."""
    data_path: str
    target_column: Optional[str]
    data: Optional[pd.DataFrame]
    data_summary: Optional[dict]
    problem_type: Optional[str]  # 'classification' or 'regression'
    use_automl: Optional[bool]  # Whether to use AutoGluon or simpler approach
    automl_config: Optional[dict]  # Configuration for AutoGluon (models, hyperparameters, etc.)
    selected_models: Optional[list[str]]  # Models for simple approach
    reasoning: Optional[str]  # LLM reasoning from subagents
    data_analysis_reasoning: Optional[str]  # Reasoning from data analysis subagent, hagat about l data
    model_selection_reasoning: Optional[str]  # Reasoning from model selection subagent, why the agent chose this model
    trained_model: Optional[Any]
    model_metrics: Optional[dict]
    error: Optional[str]#store error msgs bdl ma y crash
    step: str  # Current step in the workflow, tracks the progress of the workflow
    agent_messages: Optional[list] #store msgs excganged between sub agents
    human_approved: Optional[bool] # Conversation history from subagents


class AutoMLAgent:
    """LangGraph agent for automated machine learning model selection and training."""
    
    def __init__(self, model_name: str = None):
       
        self.model_name = model_name
        model_name = "gemini-2.5-flash"
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3,
        )
        
        self.data_analyzer = DataAnalyzer()
        self.graph = self._build_graph()
    
    
        
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph AutoML workflow."""
        
        workflow = StateGraph[AgentState, None, AgentState, AgentState](AgentState)

        # Main orchestrator nodes
        workflow.add_node("load_data", self.load_data_node)
        workflow.add_node("identify_target", self.identify_target_node)

        # Specialized subagents
        workflow.add_node("data_analysis_agent", self.data_analysis_agent)
        workflow.add_node("model_selection_agent", self.model_selection_agent)
        workflow.add_node("training_agent", self.training_agent)

        # Define the flow
        workflow.set_entry_point("load_data")#start by loading data, defines the init state
        workflow.add_edge("load_data", "identify_target")#then identify the target column
        workflow.add_edge("identify_target", "data_analysis_agent")#then analyze the data
        workflow.add_edge("data_analysis_agent", "model_selection_agent")#then select the models
        workflow.add_edge("model_selection_agent", "training_agent")#then train the models

        return workflow.compile()

    
    def should_use_automl(self, state: AgentState) -> Literal["automl", "simple"]:
        """Conditional routing function to determine next step."""
        use_automl = state.get('use_automl', False)
        return "automl" if use_automl else "simple"
    
    def load_data_node(self, state: AgentState) -> AgentState:
        """Load data from file path."""
        try:
            logger.info(f"Loading data from: {state['data_path']}")
            data = self.data_analyzer.load_data(state['data_path'])
            
            state['data'] = data
            state['step'] = 'data_loaded'
            logger.info(f"Data loaded successfully. Shape: {data.shape}")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}", e)
            state['error'] = f"Failed to load data: {str(e)}"
            state['step'] = 'error'
        
        return state
    
    def data_analysis_agent(self, state: AgentState) -> AgentState:
        """
        Deep Agent: Data Analysis Subagent
        Uses LLM reasoning to understand dataset characteristics and generate insights.
        """
        try:
            logger.info("[Data Analysis Agent] Starting deep data analysis with LLM reasoning")
            
            # First, get basic data summary
            data_summary = self.data_analyzer.get_data_summary()
            state['data_summary'] = data_summary
            
            # Prepare context for LLM reasoning
            data = state['data']
            target_col = state['target_column']
            problem_type = state['problem_type']
            
            # Create comprehensive prompt for LLM to reason about the data
            analysis_prompt = f"""
You are a senior data scientist specializing in exploratory data analysis for machine learning.

**Dataset Overview:**
- Shape: {data.shape[0]} rows, {data.shape[1]} columns
- Target column: {target_col}
- Problem type: {problem_type}
- Memory usage: {data_summary.get('data_info', {}).get('memory_mb', 0):.2f} MB

**Data Characteristics:**
- Numeric features: {data_summary.get('feature_info', {}).get('numeric_features', 0)}
- Categorical features: {data_summary.get('feature_info', {}).get('categorical_features', 0)}
- Total features: {data_summary.get('feature_info', {}).get('total_features', 0)}
- Missing values: {'Yes' if data_summary.get('data_quality', {}).get('has_missing', False) else 'No'}
- Missing percentage: {data_summary.get('data_quality', {}).get('missing_values_pct', 0):.2f}%

**Target Information:**
- Unique values: {data_summary.get('target_info', {}).get('unique_values', 0)}
"""
            
            if problem_type == 'classification':
                class_dist = data_summary.get('target_info', {}).get('class_distribution', {})
                analysis_prompt += f"- Class distribution: {class_dist}\n"
            else:
                target_stats = data_summary.get('target_info', {}).get('statistics', {})
                analysis_prompt += f"- Target statistics: {target_stats}\n"
            
            analysis_prompt += """
**Your Task:**
Provide a comprehensive analysis of this dataset focusing on:
1. Data quality assessment (missing values, outliers, data types)
2. Feature characteristics (distributions, relationships, importance)
3. Complexity assessment (is this a simple or complex problem?)
4. Potential challenges for ML models
5. Recommendations for preprocessing steps needed

Provide your analysis in a clear, structured format:
"""
            
            # Get LLM reasoning about the data
            messages = [
                SystemMessage(content="You are an expert data scientist. Provide detailed, actionable analysis of datasets for machine learning projects. Focus on practical insights that inform model selection."),
                HumanMessage(content=analysis_prompt)
            ]
            
            try:
                response = self.llm.invoke(messages)
                analysis_reasoning = response.content
                
                # Update agent messages for context
                if 'agent_messages' not in state:
                    state['agent_messages'] = []
                state['agent_messages'].append({
                    'agent': 'data_analysis',
                    'message': analysis_reasoning
                })
                
                state['data_analysis_reasoning'] = analysis_reasoning
                state['reasoning'] = analysis_reasoning
                state['step'] = 'data_analyzed'
                
                logger.info(f"[Data Analysis Agent] Analysis complete. Reasoning length: {len(analysis_reasoning)} chars")
                logger.info(f"[Data Analysis Agent] Key insights: {analysis_reasoning[:200]}...")
                
            except Exception as e:
                logger.warn(f"[Data Analysis Agent] LLM reasoning failed: {str(e)}, using basic analysis...")
                state['data_analysis_reasoning'] = f"Basic analysis: Dataset has {data.shape[0]} rows, {data.shape[1]} columns. Problem type: {problem_type}"
                state['step'] = 'data_analyzed'
            
        except Exception as e:
            logger.error(f"[Data Analysis Agent] Error: {str(e)}", e)
            state['error'] = f"Failed in data analysis: {str(e)}"
            state['step'] = 'error'
        
        return state
    
    def identify_target_node(self, state: AgentState) -> AgentState:
        """Identify or validate the target column."""
        try:
            logger.info("Identifying target column")
            target_col = state.get('target_column')
            
            if target_col:
                self.data_analyzer.identify_target_column(target_col)
            else:
                target_col = self.data_analyzer.identify_target_column()
            
            problem_type = self.data_analyzer.determine_problem_type()
            
            state['target_column'] = target_col
            state['problem_type'] = problem_type
            state['step'] = 'target_identified'
            logger.info(f"Target column identified: {target_col}, Problem type: {problem_type}")
            
        except Exception as e:
            logger.error(f"Error identifying target: {str(e)}", e)
            state['error'] = f"Failed to identify target: {str(e)}"
            state['step'] = 'error'
        
        return state
    
    def model_selection_agent(self, state: AgentState) -> AgentState:
        """
        Deep Agent: Model Selection Subagent
        Uses LLM planning and reasoning to decide on AutoML vs simple training and select models.
        """
        try:
            logger.info("[Model Selection Agent] Starting  model selection with LLM planning")
            
            data_summary = state['data_summary']
            problem_type = state['problem_type']
            data_analysis_reasoning = state.get('data_analysis_reasoning', '')
            
            # Create enhanced prompt with data analysis context
            prompt = self._create_automl_decision_prompt(data_summary, problem_type, data_analysis_reasoning)
            
            # Get LLM reasoning with planning
            messages = [
                SystemMessage(content="You are a senior ML architect specializing in automated ML and model selection. You plan the best strategy considering complexity, resources, and performance requirements. Think step-by-step and provide detailed reasoning."),
                HumanMessage(content=prompt)
            ]
            
            # Include previous agent context if available
            if data_analysis_reasoning:
                messages.insert(1, AIMessage(content=f"Data Analysis Agent's findings:\n{data_analysis_reasoning}"))
            
            try:
                response = self.llm.invoke(messages)
                reasoning = response.content
                
                # Update agent messages
                if 'agent_messages' not in state:
                    state['agent_messages'] = []
                state['agent_messages'].append({
                    'agent': 'model_selection',
                    'message': reasoning
                })
                
                # Parse LLM decision
                use_automl, automl_config, selected_models = self._parse_automl_decision(
                    reasoning, data_summary, problem_type
                )
                
                state['model_selection_reasoning'] = reasoning
                state['reasoning'] = reasoning
                state['use_automl'] = use_automl
                state['automl_config'] = automl_config
                state['selected_models'] = selected_models
                state['step'] = 'models_selected'
                
                if use_automl:
                    logger.info(f"[Model Selection Agent] Decision: Using AutoGluon with config: {automl_config}")
                    logger.info(f"[Model Selection Agent] Selected models: {automl_config.get('models', [])}")
                else:
                    logger.info(f"[Model Selection Agent] Decision: Using simple approach with models: {selected_models}")
                logger.info(f"[Model Selection Agent] Reasoning preview: {reasoning[:300]}...")
                
            except Exception as e:
                logger.warn(f"[Model Selection Agent] LLM call failed: {str(e)}, using heuristic-based fallback...")
                # Fallback: Use heuristics to decide
                rows = data_summary.get('data_info', {}).get('rows', 0)
                features = data_summary.get('feature_info', {}).get('total_features', 0)
                has_missing = data_summary.get('data_quality', {}).get('has_missing', False)
                
                # Heuristic decision
                use_automl = rows > 10000 or features > 20 or has_missing
                
                if use_automl:
                    automl_config = {
                        'models': ['GBM', 'XGBoost', 'LightGBM'],
                        'time_limit': 300,
                        'preset': 'best_quality'
                    }
                    selected_models = []
                else:
                    automl_config = {}
                    default_models = {
                        'classification': ['RandomForest', 'GradientBoosting'],
                        'regression': ['RandomForest', 'GradientBoosting']
                    }
                    selected_models = default_models.get(problem_type, ['RandomForest'])
                
                state['model_selection_reasoning'] = f"LLM unavailable, using heuristic: dataset has {rows} rows, {features} features. Decision: {'AutoGluon' if use_automl else 'Simple training'}"
                state['reasoning'] = state['model_selection_reasoning']
                state['use_automl'] = use_automl
                state['automl_config'] = automl_config
                state['selected_models'] = selected_models
                state['step'] = 'models_selected'
                
                logger.info(f"[Model Selection Agent] Fallback decision: Using {'AutoGluon' if use_automl else 'Simple'} approach with models: {selected_models}")
        
        except Exception as e:
            logger.error(f"[Model Selection Agent] Error: {str(e)}", e)
            state['error'] = f"Failed in model selection: {str(e)}"
            state['step'] = 'error'
        
        return state
   

    def training_agent(self, state: AgentState) -> AgentState:
        """
        Deep Agent: Training Subagent with Confusion Matrix Support.
        """
        try:
            use_automl = state.get('use_automl', False)
            model_selection_reasoning = state.get('model_selection_reasoning', '')
            
            if use_automl:
                logger.info("[Training Agent] Executing AutoGluon training strategy")
            else:
                logger.info("[Training Agent] Executing simple training strategy")
            
            # --- LLM strategy assessment block (kept your style) ---
            try:
                training_strategy_prompt = f"Assess this strategy: {'AutoGluon' if use_automl else 'Simple'}. Reasoning: {model_selection_reasoning[:500]}"
                strategy_messages = [
                    SystemMessage(content="You are an ML engineer. Provide brief insights."),
                    HumanMessage(content=training_strategy_prompt)
                ]
                strategy_response = self.llm.invoke(strategy_messages)
                training_insight = strategy_response.content
            except Exception as e:
                training_insight = "Executing training strategy..."
            
            # Execute training
            data = state['data']
            target_column = state['target_column']
            problem_type = state['problem_type']
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            # Split data here to ensure we have test labels for the confusion matrix
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import confusion_matrix
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if use_automl:
                automl_config = state.get('automl_config', {})
                trained_model, metrics = self._train_with_autogluon(X_train, y_train, problem_type, automl_config)
                # AutoGluon needs its own internal test data format
                y_pred = trained_model.predict(X_test)
            else:
                selected_models = state.get('selected_models', []) or ['RandomForest']
                trained_model, metrics = self._train_simple_models(X, y, problem_type, selected_models)
                # Simple models use scikit-learn predict
                X_test_simple = pd.get_dummies(X_test, drop_first=True)
                # Re-align columns in case dummies changed
                X_test_simple = X_test_simple.reindex(columns=pd.get_dummies(X_train, drop_first=True).columns, fill_value=0)
                y_pred = trained_model.predict(X_test_simple)
            
            # --- FIX: ADD CONFUSION MATRIX ---
            if problem_type == 'classification':
                cm = confusion_matrix(y_test, y_pred)
                metrics['confusion_matrix'] = cm.tolist() # Save as list for JSON/Markdown compatibility
                logger.info(f"[Training Agent] Confusion Matrix generated: {metrics['confusion_matrix']}")

            # Get LLM interpretation of results
            results_interpretation = self._interpret_training_results(metrics, use_automl)
            
            # Update state
            state['trained_model'] = trained_model
            state['model_metrics'] = metrics
            state['step'] = 'model_trained'
            
            if 'agent_messages' not in state:
                state['agent_messages'] = []
            state['agent_messages'].append({
                'agent': 'training',
                'message': f"Training complete. {results_interpretation}"
            })
            
            logger.info(f"[Training Agent] Training complete. Score: {metrics.get('best_score', 0):.4f}")
            
        except Exception as e:
            logger.error(f"[Training Agent] Error: {str(e)}", e)
            state['error'] = f"Failed to train model: {str(e)}"
            state['step'] = 'error'
        
        return state

    def _interpret_training_results(self, metrics: dict, use_automl: bool) -> str:
        """Use LLM to interpret training results including Confusion Matrix."""
        try:
            # Prepare Confusion Matrix text if it exists
            cm_text = ""
            if 'confusion_matrix' in metrics:
                cm = metrics['confusion_matrix']
                cm_text = f"\n**Confusion Matrix:**\nPredicted 0: [{cm[0][0]}, {cm[0][1]}]\nPredicted 1: [{cm[1][0]}, {cm[1][1]}]"

            interpretation_prompt = f"""
Analyze the following model training results:

**Training Method:** {'AutoGluon AutoML' if use_automl else 'Simple Direct Training'}
**Best Model:** {metrics.get('best_model', 'N/A')}
**Best Score:** {metrics.get('best_score', 0):.4f}
**Models Trained:** {metrics.get('models_trained', 0)}
**All Models:** {metrics.get('all_models', [])}
{cm_text}

Provide a brief interpretation:
1. Performance assessment (excellent/good/fair/poor)
2. Based on the Confusion Matrix, are there more False Positives or False Negatives?
3. Any recommendations for improvement

Keep it concise (3-4 sentences).
"""
            
            messages = [
                SystemMessage(content="You are an ML expert interpreting model training results. Provide clear, actionable insights."),
                HumanMessage(content=interpretation_prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            logger.warn(f"Could not get LLM interpretation: {str(e)}")
            return f"Training completed. Best score: {metrics.get('best_score', 0):.4f}"
    
    def _create_automl_decision_prompt(self, data_summary: dict, problem_type: str, data_analysis_reasoning: str = '') -> str:
        """Create a prompt for LLM to decide on AutoML usage and configuration."""
        prompt = f"""
Analyze the following tabular dataset characteristics and decide the best approach for this machine learning problem.

**Problem Type:** {problem_type}

**Dataset Characteristics:**
- Number of rows: {data_summary['data_info']['rows']}
- Number of features: {data_summary['feature_info']['total_features']}
- Numeric features: {data_summary['feature_info']['numeric_features']}
- Categorical features: {data_summary['feature_info']['categorical_features']}
- Missing values: {data_summary['data_quality']['has_missing']}
- Missing values percentage: {data_summary['data_quality']['missing_values_pct']}%
- Memory usage: {data_summary['data_info']['memory_mb']} MB

**Target Information:**
- Target column: {data_summary['target_info']['column']}
- Unique values: {data_summary['target_info']['unique_values']}
"""
        
        if problem_type == 'classification':
            class_dist = data_summary['target_info'].get('class_distribution', {})
            prompt += f"- Class distribution: {class_dist}\n"
        
        if data_analysis_reasoning:
            prompt += f"""
**Data Analysis Agent's Insights:**
{data_analysis_reasoning}

"""
        
        prompt += """
**Decision Task:**

As the Model Selection Agent, you need to plan and decide between two approaches:

1. **AutoGluon (AutoML Framework)**: Best for complex problems, large datasets, or when high performance is critical.
   - If you choose this, you MUST specify the models to prioritize, a time limit, and a preset.
   - Valid Models: ['GBM', 'CAT', 'XGB', 'RF', 'XT', 'KNN', 'LR', 'NN_TORCH', 'FASTAI']

2. **Simple Direct Training**: Best for small datasets or when simple, interpretable models are preferred.
   - If you choose this, you MUST suggest exactly 3 specific models to train for comparison.
   - Valid Models: ['RandomForest', 'XGBoost', 'GradientBoosting', 'LogisticRegression', 'LinearRegression']

**Response Format (STRICT JSON):**
Your response must be a single JSON object with these EXACT keys:
{
  "approach": "AutoGluon" or "Simple",
  "reasoning": "Detailed explanation of your choice",
  "autogluon_settings": {
    "models_to_prioritize": ["GBM", "XGB", "CAT"],
    "time_limit_seconds": 300,
    "preset_mode": "best_quality"
  },
  "simple_models": ["Model1", "Model2", "Model3"]
}

Provide your analysis and decision:
"""
        return prompt
    
    def _parse_automl_decision(self, reasoning: str, data_summary: dict, problem_type: str) -> tuple:
        """
        Parse LLM reasoning to extract AutoML decision and configuration.
        
        Returns:
            tuple: (use_automl: bool, automl_config: dict, selected_models: list)
        """
        import re
        import json
        
        reasoning_lower = reasoning.lower()
        use_automl = None
        automl_config = {}
        selected_models = []

        # PRIORITY 1: Try to parse structured JSON from the reasoning
        try:
            json_match = re.search(r'\{.*\}', reasoning, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                
                # Check approach key (matches the new prompt)
                approach = str(data.get("approach") or "").lower()
                if "autogluon" in approach:
                    use_automl = True
                elif "simple" in approach:
                    use_automl = False
                
                # If JSON explicitly decided, extract the rest
                if use_automl is not None:
                    if use_automl:
                        settings = data.get("autogluon_settings", {})
                        automl_config = {
                            'models': settings.get("models_to_prioritize", ['GBM', 'XGB', 'CAT']),
                            'time_limit': settings.get("time_limit_seconds", 300),
                            'preset': settings.get("preset_mode", 'best_quality')
                        }
                    else:
                        selected_models = data.get("simple_models", [])
        except Exception as e:
            logger.warn(f"JSON parsing failed, falling back to regex: {e}")

        # PRIORITY 2: Regex Fallback (Your original style)
        if use_automl is None:
            if re.search(r'use_automl\s*[:=]\s*(false|0|no)', reasoning_lower):
                use_automl = False
            elif re.search(r'use_automl\s*[:=]\s*(true|1|yes)', reasoning_lower):
                use_automl = True
        
        # PRIORITY 3: Heuristics (Your original style)
        if use_automl is None:
            rows = data_summary['data_info']['rows']
            features = data_summary['feature_info']['total_features']
            use_automl = rows > 10000 or features > 20

        # Final Config Assembly
        if use_automl:
            # If JSON didn't provide config, use your defaults + regex extraction
            if not automl_config:
                automl_config = {
                    'models': ['GBM', 'XGBoost', 'LightGBM', 'CatBoost'],
                    'time_limit': 300,
                    'preset': 'best_quality'
                }
                
                models_match = re.search(r'models\s*[:=]\s*\[([^\]]+)\]', reasoning, re.IGNORECASE)
                if models_match:
                    models_str = models_match.group(1)
                    models = [m.strip().strip("'\"") for m in models_str.split(',')]
                    valid_models = ['GBM', 'XGBoost', 'LightGBM', 'CatBoost', 'NeuralNet', 'FastAI', 'RF', 'XT', 'KNN', 'LR']
                    automl_config['models'] = [m for m in models if m in valid_models][:8]

                time_match = re.search(r'time_limit\s*[:=]\s*(\d+)', reasoning, re.IGNORECASE)
                if time_match:
                    automl_config['time_limit'] = int(time_match.group(1))

                preset_match = re.search(r'preset\s*[:=]\s*([^\s,\]]+)', reasoning, re.IGNORECASE)
                if preset_match:
                    preset = preset_match.group(1).strip("'\"")
                    valid_presets = ['best_quality', 'high_quality', 'good_quality_faster_inference', 'optimize_for_deployment']
                    if preset in valid_presets:
                        automl_config['preset'] = preset
        else:
            # If JSON didn't provide models, use your regex extraction + defaults
            if not selected_models:
                models_match = re.search(r'simple_models\s*[:=]\s*\[([^\]]+)\]', reasoning, re.IGNORECASE)
                if models_match:
                    models_str = models_match.group(1)
                    selected_models = [m.strip().strip("'\"") for m in models_str.split(',')]
                
                if not selected_models:
                    default_models = {
                        'classification': ['RandomForest', 'GradientBoosting'],
                        'regression': ['RandomForest', 'GradientBoosting']
                    }
                    selected_models = default_models.get(problem_type, ['RandomForest'])
        
        return use_automl, automl_config, selected_models[:3]
    def _train_with_autogluon(self, X: pd.DataFrame, y: pd.Series, problem_type: str, config: dict) -> tuple:
        """
        Train model using AutoGluon AutoML framework with LLM-recommended configuration.
        AutoGluon will automatically select the best model after training.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            problem_type: 'classification' or 'regression'
            config: Dictionary with 'models', 'time_limit', 'preset' keys
        
        Returns:
            tuple: (trained_model, metrics_dict)
        """
        try:
            from autogluon.tabular import TabularPredictor
            
            logger.info(f"Initializing AutoGluon predictor with config: {config}")
            
            # Prepare data with target column
            train_data = X.copy()
            target_col_name = 'target'
            train_data[target_col_name] = y
            
            # Map problem type to AutoGluon's expected format
            # AutoGluon expects 'binary', 'multiclass', or 'regression'
            ag_problem_type = problem_type
            if problem_type == 'classification':
                # Determine if binary or multiclass based on target values
                unique_targets = y.nunique()
                if unique_targets == 2:
                    ag_problem_type = 'binary'
                else:
                    ag_problem_type = 'multiclass'
                logger.info(f"Mapped 'classification' to '{ag_problem_type}' ({unique_targets} classes)")
            
            # Create a unique path for this predictor to avoid conflicts
            import tempfile
            predictor_path = os.path.join(tempfile.gettempdir(), f"autogluon_predictor_{os.getpid()}_{int(time.time())}")
            
            # Create predictor (AutoGluon will auto-select the best metric if not specified)
            predictor = TabularPredictor(
                label=target_col_name,
                problem_type=ag_problem_type,
                path=predictor_path
            )
            
            # Extract configuration
            models = config.get('models', ['GBM', 'XGBoost', 'LightGBM'])
            time_limit = config.get('time_limit', 300)
            preset = config.get('preset', 'best_quality')
            
            logger.info(f"Training AutoGluon with models: {models}, time_limit: {time_limit}s, preset: {preset}...")
            
            # Map model names to AutoGluon model types
            # AutoGluon uses specific model type names: 'RF', 'XT', 'KNN', 'GBM', 'CAT', 'XGB', 'NN_TORCH', 'LR', 'FASTAI', etc.
            model_type_mapping = {
                'GBM': 'GBM',
                'XGBoost': 'XGB',  # AutoGluon uses 'XGB' not 'XGBoost'
                'LightGBM': 'GBM',  # LightGBM functionality is included in GBM
                'CatBoost': 'CAT',
                'NeuralNet': 'NN_TORCH',
                'NeuralNetwork': 'NN_TORCH',
                'FastAI': 'FASTAI',
                'RandomForest': 'RF',
                'RF': 'RF',
                'ExtraTrees': 'XT',
                'XT': 'XT',
                'KNN': 'KNN',
                'LR': 'LR',
                'LinearModel': 'LR',
                'LinearRegression': 'LR',
                'LogisticRegression': 'LR'
            }
            
            # Convert model names to AutoGluon model types
            ag_models = []
            for m in models:
                mapped = model_type_mapping.get(m, None)
                if mapped:
                    if mapped not in ag_models:  # Avoid duplicates
                        ag_models.append(mapped)
                else:
                    logger.warn(f"Unknown model '{m}' - skipping. Valid AutoGluon models: {list(model_type_mapping.values())}")
            
            # Build hyperparameters dict for selected models
            hyperparameters = {}
            if ag_models:
                # Only include models that are in the list
                # AutoGluon hyperparameters format: {'MODEL_TYPE': {}}
                for model_type in ag_models:
                    hyperparameters[model_type] = {}
            
            # Train with specified configuration
            # Note: AutoGluon will automatically select the best model after training all specified models
            fit_kwargs = {
                'time_limit': time_limit,
                'presets': preset
            }
            
            if hyperparameters:
                fit_kwargs['hyperparameters'] = hyperparameters
            
            predictor.fit(train_data, **fit_kwargs)
            
            # Get leaderboard to see all models and their performance
            leaderboard = predictor.leaderboard(silent=True)
            
            # AutoGluon automatically selects the best model (first row of leaderboard)
            # Get information about the best model
            if len(leaderboard) > 0:
                best_model_name = leaderboard.iloc[0]['model']
                best_score = float(leaderboard.iloc[0]['score_val'])
            else:
                best_model_name = "Unknown"
                best_score = 0.0
            
            # Extract metrics
            metrics = {
                'best_model': best_model_name,
                'best_score': best_score,
                'models_trained': len(leaderboard),
                'all_models': leaderboard['model'].tolist() if len(leaderboard) > 0 else [],
                'all_scores': leaderboard['score_val'].tolist() if len(leaderboard) > 0 else [],
                'training_method': 'AutoGluon'
            }
            
            logger.info(f"AutoGluon training complete. Best model: {metrics['best_model']} (score: {metrics['best_score']:.4f})")
            logger.info(f"Total models trained: {metrics['models_trained']}")
            
            return predictor, metrics
            
        except ImportError:
            logger.warn("AutoGluon not available, falling back to simple training...")
            # Fallback to simple training with first recommended model
            fallback_models = config.get('models', ['RandomForest'])
            return self._train_simple_models(X, y, problem_type, [fallback_models[0]] if fallback_models else ['RandomForest'])
        except Exception as e:
            logger.warn(f"AutoGluon training failed: {str(e)}, falling back to simple training...")
            # Fallback to simple training
            fallback_models = config.get('models', ['RandomForest'])
            return self._train_simple_models(X, y, problem_type, [fallback_models[0]] if fallback_models else ['RandomForest'])
    
    def _train_simple_models(self, X: pd.DataFrame, y: pd.Series, problem_type: str, model_names: list[str]) -> tuple:
        """
        Train models using simple scikit-learn approach with LLM-selected models.
        """
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.metrics import accuracy_score, r2_score, f1_score, mean_squared_error
        
        try:
            import xgboost as xgb
            xgb_available = True
        except ImportError:
            xgb_available = False
            logger.warn("XGBoost not available, will use GradientBoosting as alternative")
        
        # Ensure we have a valid list to iterate over
        if not model_names or not isinstance(model_names, list):
            model_names = ['RandomForest', 'GradientBoosting'] if problem_type == 'classification' else ['RandomForest']
            
        logger.info(f"Training simple models: {model_names}")
        
        X_processed = pd.get_dummies(X, drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )
        
        best_model = None
        best_score = -float('inf')
        best_model_name = None
        best_metric_name = 'accuracy' if problem_type == 'classification' else 'r2_score'
        all_results = []
        
        for model_name in model_names:
            try:
                if problem_type == 'classification':
                    if 'RandomForest' in model_name or 'randomforest' in model_name.lower():
                        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                    elif ('XGBoost' in model_name or 'xgboost' in model_name.lower() or 'xgb' in model_name.lower()) and xgb_available:
                        model = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
                    elif 'GradientBoosting' in model_name or 'gradient' in model_name.lower():
                        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
                    elif 'LogisticRegression' in model_name or 'logistic' in model_name.lower():
                        model = LogisticRegression(max_iter=1000, random_state=42)
                    else:
                        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                    
                    model.fit(X_train, y_train)
                    score = accuracy_score(y_test, model.predict(X_test))
                    
                else:  # regression
                    if 'RandomForest' in model_name or 'randomforest' in model_name.lower():
                        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                    elif ('XGBoost' in model_name or 'xgboost' in model_name.lower() or 'xgb' in model_name.lower()) and xgb_available:
                        model = xgb.XGBRegressor(n_estimators=100, random_state=42)
                    elif 'GradientBoosting' in model_name or 'gradient' in model_name.lower():
                        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                    elif 'LinearRegression' in model_name or 'linear' in model_name.lower():
                        model = LinearRegression()
                    else:
                        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                    
                    model.fit(X_train, y_train)
                    score = r2_score(y_test, model.predict(X_test))
                
                all_results.append({
                    'model_name': model_name,
                    'score': float(score)
                })
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_model_name = model_name
                
                logger.info(f"{model_name} - {best_metric_name}: {score:.4f}")
                
            except Exception as e:
                logger.warn(f"Failed to train {model_name}: {str(e)}")
                continue
        
        # Fallback if the loop didn't produce a model
        if best_model is None:
            logger.warn("All selected models failed, using default RandomForest...")
            best_model_name = 'RandomForest'
            best_model = RandomForestClassifier() if problem_type == 'classification' else RandomForestRegressor()
            best_model.fit(X_train, y_train)
            best_score = best_model.score(X_test, y_test)
            all_results = [{'model_name': 'RandomForest', 'score': float(best_score)}]

        metrics = {
            'best_model': best_model_name,
            'best_score': float(best_score),
            'metric_name': best_metric_name,
            'models_trained': len(all_results),
            'all_models': [r['model_name'] for r in all_results],
            'all_scores': [r['score'] for r in all_results],
            'training_method': 'Simple'
        }
        
        logger.info(f"Best model: {metrics['best_model']} with {best_metric_name}: {best_score:.4f}")
        
        return best_model, metrics
    
    def run(self, data_path: str, target_column: str = None) -> dict:
        """
        Run the complete AutoML workflow.
        
        Args:
            data_path: Path to the data file
            target_column: Optional target column name (will be auto-detected if not provided)
        
        Returns:
            dict: Final state with results
        """
        initial_state = {
            'data_path': data_path,
            'target_column': target_column,
            'data': None,
            'data_summary': None,
            'problem_type': None,
            'use_automl': None,
            'automl_config': None,
            'selected_models': None,
            'reasoning': None,
            'data_analysis_reasoning': None,
            'model_selection_reasoning': None,
            'trained_model': None,
            'model_metrics': None,
            'error': None,
            'step': 'initialized',
            'agent_messages': []
        }
        
        logger.info("Starting AutoML agent workflow")
        final_state = self.graph.invoke(initial_state)
        
        if final_state.get('error'):
            logger.error(f"Workflow completed with error: {final_state['error']}")
        else:
            logger.info("Workflow completed successfully!")
        
        return final_state

