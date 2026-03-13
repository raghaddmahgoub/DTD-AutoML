
import json
from typing import TypedDict, Annotated, Literal, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
import time
from dotenv import load_dotenv

import dask.dataframe as dd

from src.utils.logger import Logger
# Load environment variables
import pickle
from datetime import datetime
import json
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
    # data_analysis_reasoning: Optional[str]  # Reasoning from data analysis subagent, hagat about l data
    automl_directives: Optional[dict]  # =====> Add this to receive external analysis
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
        
        
        self.graph = self._build_graph()
    
    
        
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph AutoML workflow."""
        
        workflow = StateGraph[AgentState, None, AgentState, AgentState](AgentState)

        # Main orchestrator nodes
        workflow.add_node("load_data", self.load_data_node)
        workflow.add_node("identify_target", self.identify_target_node)

        # Specialized subagents
        # workflow.add_node("data_analysis_agent", self.data_analysis_agent)
        workflow.add_node("model_selection_agent", self.model_selection_agent)
        workflow.add_node("training_agent", self.training_agent)

        # Define the flow
        workflow.set_entry_point("load_data")#start by loading data, defines the init state
        workflow.add_edge("load_data", "identify_target")#then identify the target column
        # workflow.add_edge("identify_target", "data_analysis_agent")#then analyze the data
        # workflow.add_edge("data_analysis_agent", "model_selection_agent")#then select the models
        workflow.add_edge("identify_target", "model_selection_agent")#then select the models
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
            data_path = str(state['data_path'])
            if data_path.endswith('.csv'):
                self.data = dd.read_csv(
                                data_path,
                                blocksize="256MB",
                                assume_missing=True,
                                dtype=str,
                                sample=256000
                            )
            elif data_path.endswith('.xlsx') or data_path.endswith('.xls'):
                pdf = pd.read_excel(data_path)
                self.data = dd.from_pandas(pdf, npartitions=4)
            elif data_path.endswith('.json'):
                self.data = dd.read_json(data_path, blocksize="256MB", lines=True)
            else:
                raise ValueError(f"Unsupported file format")
            
            state['data'] = self.data
            state['step'] = 'data_loaded'
            logger.info(f"Data loaded successfully. Shape: {len(self.data.columns)}")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}", e)
            state['error'] = f"Failed to load data: {str(e)}"
            state['step'] = 'error'
        
        return state
    

    def identify_target_node(self, state: AgentState) -> AgentState:
        try:
            logger.info("Identifying target column")
            target_col = state.get('target_column')
            
            if not target_col:
                target_col = state['data'].columns[-1]
                logger.info(f"Target not provided, falling back to last column: {target_col}")

            # ── FIX: only auto-detect problem type if not already set by orchestrator ──
            problem_type = state.get('problem_type')
            if not problem_type:
                target_data = state['data'][target_col]
                if pd.api.types.is_numeric_dtype(target_data) and target_data.nunique() > 20:
                    problem_type = 'regression'
                else:
                    problem_type = 'classification'
                logger.info(f"Problem type auto-detected: {problem_type}")
            else:
                logger.info(f"Problem type pre-set by orchestrator: {problem_type}")

            state['target_column'] = target_col
            state['problem_type']  = problem_type
            state['step']          = 'target_identified'
            logger.info(f"Target column: {target_col}, Problem type: {problem_type}")

        except Exception as e:
            logger.error(f"Error identifying target: {str(e)}", e)
            state['error'] = f"Failed to identify target: {str(e)}"
            state['step']  = 'error'

        return state
    
    # def model_selection_agent(self, state: AgentState) -> AgentState:
    #     """
    #     Deep Agent: Model Selection Subagent
    #     Uses LLM planning and reasoning to decide on AutoML vs simple training and select models.
    #     """
    #     try:
    #         logger.info("[Model Selection Agent] Starting  model selection with LLM planning")
            
    #         data_summary = state['data_summary']
    #         problem_type = state['problem_type']
    #         data_analysis_reasoning = state.get('data_analysis_reasoning', '')
    #         # Create enhanced prompt with data analysis context
    #         prompt = self._create_automl_decision_prompt(data_summary, problem_type, data_analysis_reasoning)
            
    #         # Get LLM reasoning with planning
    #         messages = [
    #             SystemMessage(content="You are a senior ML architect specializing in automated ML and model selection. You plan the best strategy considering complexity, resources, and performance requirements. Think step-by-step and provide detailed reasoning."),
    #             HumanMessage(content=prompt)
    #         ]
            
    #         # Include previous agent context if available
    #         if data_analysis_reasoning:
    #             messages.insert(1, AIMessage(content=f"Data Analysis Agent's findings:\n{data_analysis_reasoning}"))
            
    #         try:
    #             response = self.llm.invoke(messages)
    #             reasoning = response.content
                
    #             # Update agent messages
    #             if 'agent_messages' not in state:
    #                 state['agent_messages'] = []
    #             state['agent_messages'].append({
    #                 'agent': 'model_selection',
    #                 'message': reasoning
    #             })
                
    #             # Parse LLM decision
    #             use_automl, automl_config, selected_models = self._parse_automl_decision(
    #                 reasoning, data_summary, problem_type
    #             )
                
    #             state['model_selection_reasoning'] = reasoning
    #             state['reasoning'] = reasoning
    #             state['use_automl'] = use_automl
    #             state['automl_config'] = automl_config
    #             state['selected_models'] = selected_models
    #             state['step'] = 'models_selected'
                
    #             if use_automl:
    #                 logger.info(f"[Model Selection Agent] Decision: Using AutoGluon with config: {automl_config}")
    #                 logger.info(f"[Model Selection Agent] Selected models: {automl_config.get('models', [])}")
    #             else:
    #                 logger.info(f"[Model Selection Agent] Decision: Using simple approach with models: {selected_models}")
    #             logger.info(f"[Model Selection Agent] Reasoning preview: {reasoning[:300]}...")
                
    #         except Exception as e:
    #             logger.warn(f"[Model Selection Agent] LLM call failed: {str(e)}, using heuristic-based fallback...")
    #             # Fallback: Use heuristics to decide
    #             rows = data_summary.get('data_info', {}).get('rows', 0)
    #             features = data_summary.get('feature_info', {}).get('total_features', 0)
    #             has_missing = data_summary.get('data_quality', {}).get('has_missing', False)
                
    #             # Heuristic decision
    #             use_automl = rows > 10000 or features > 20 or has_missing
                
    #             if use_automl:
    #                 automl_config = {
    #                     'models': ['GBM', 'XGBoost', 'LightGBM'],
    #                     'time_limit': 300,
    #                     'preset': 'best_quality'
    #                 }
    #                 selected_models = []
    #             else:
    #                 automl_config = {}
    #                 default_models = {
    #                     'classification': ['RandomForest', 'GradientBoosting'],
    #                     'regression': ['RandomForest', 'GradientBoosting']
    #                 }
    #                 selected_models = default_models.get(problem_type, ['RandomForest'])
                
    #             state['model_selection_reasoning'] = f"LLM unavailable, using heuristic: dataset has {rows} rows, {features} features. Decision: {'AutoGluon' if use_automl else 'Simple training'}"
    #             state['reasoning'] = state['model_selection_reasoning']
    #             state['use_automl'] = use_automl
    #             state['automl_config'] = automl_config
    #             state['selected_models'] = selected_models
    #             state['step'] = 'models_selected'
                
    #             logger.info(f"[Model Selection Agent] Fallback decision: Using {'AutoGluon' if use_automl else 'Simple'} approach with models: {selected_models}")
        
    #     except Exception as e:
    #         logger.error(f"[Model Selection Agent] Error: {str(e)}", e)
    #         state['error'] = f"Failed in model selection: {str(e)}"
    #         state['step'] = 'error'
        
    #     return state
    
    def model_selection_agent(self, state: AgentState) -> AgentState:
        """
        Model Selection Subagent that depends on external Analysis Agent directives.
        """
        try:
            logger.info("[Model Selection Agent] Starting selection using external directives")
            
            # 1. Retrieve the directives from the shared state
            directives = state.get('automl_directives', {})
            task_type = state.get('problem_type') or directives.get('task_type')
            rows = state.get('data_summary', {}).get('data_info', {}).get('rows', 0)
            
            # 2. Extract key signals for the LLM prompt
            # We access the report structure generated by eda_agent2.py
            report = directives.get('report', {})
            target_info = report.get('target_analysis', {})
            encoding_hints = directives.get('encoding_hints', {})
            multicollinearity = directives.get('multicollinearity', {})
            signal_analysis = directives.get('signal_analysis', {})

            # 3. Construct a directive-aware prompt
            prompt = f"""
            Analyze these pre-calculated dataset characteristics to select the best ML strategy.
            
            **Target Information:**
            - Task Type: {task_type}
            - Target Column: {target_info.get('column')}
            - Skew Severity: {target_info.get('skew_severity', 'N/A')}
            
            **Feature Engineering Directives:**
            - Recommended Encodings: {json.dumps(encoding_hints)}
            - Multicollinearity Risk: {json.dumps(multicollinearity.get('pairs', []))}
            
            **Signal Analysis:**
            - Feature Strengths: {json.dumps(signal_analysis)}
            
            **Decision Task:**
            Plan and decide between:
            1. AutoGluon: For high complexity or non-linear signals.
            2. Simple Training: For linear signals or smaller datasets.
            
            Response Format (STRICT JSON):
            {{
            "approach": "AutoGluon" or "Simple",
            "reasoning": "Explain why based on the signals above",
            "autogluon_settings": {{ "models_to_prioritize": ["GBM", "XGB"], "time_limit_seconds": 300, "preset_mode": "best_quality" }},
            "simple_models": ["RandomForest", "XGBoost", "LogisticRegression"]
            }}
            """

            # 4. Invoke LLM with directives
            messages = [
                SystemMessage(content="You are a senior ML architect. Use the provided data analysis to choose a model."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            reasoning = response.content
            
            # 5. Parse and update state
            use_automl, automl_config, selected_models = self._parse_automl_decision(
                reasoning, state.get('data_summary', {}), task_type
            )
            
            state['model_selection_reasoning'] = reasoning
            state['use_automl'] = use_automl
            state['automl_config'] = automl_config
            state['selected_models'] = selected_models
            state['step'] = 'models_selected'
            
                        # Large-scale safeguard
            if use_automl and rows > 1_000_000:
                automl_config['models_to_prioritize'] = \
                    automl_config.get('models_to_prioritize', ['GBM', 'XGB'])[:2]
                
                automl_config['preset_mode'] = 'good_quality_faster_inference'
                
                automl_config['time_limit_seconds'] = \
                    min(automl_config.get('time_limit_seconds', 300), 600)

                state['automl_config'] = automl_config
                
            state['step'] = 'models_selected'

            return state

        except Exception as e:
            logger.error(f"[Model Selection Agent] Error: {str(e)}")
            state['error'] = f"Failed in model selection: {str(e)}"
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
            
            # 1. LOAD AND COERCE DATA TYPES
            data = state['data'].copy()
            target_column = state['target_column']
            problem_type = state['problem_type']
            
                        # Ensure Dask compatibility
            if not isinstance(data, dd.DataFrame):
                data = dd.from_pandas(data, npartitions=10)

            X = data.drop(columns=[target_column])
            y = data[target_column]

            # Convert all columns to numeric where possible
            X = X.map_partitions(lambda df: df.apply(pd.to_numeric, errors="coerce"))
            y = dd.to_numeric(y, errors="coerce")

            # Convert to pandas before sklearn/AutoGluon
            if hasattr(X, "compute"):
                X = X.compute()
            if hasattr(y, "compute"):
                y = y.compute()

            # Drop rows where target is missing
            valid_idx = ~y.isna()
            X = X.loc[valid_idx]
            y = y.loc[valid_idx]

            # # FORCE Target to numeric for Regression
            # if state['problem_type'] == 'regression':
            #     data[target_column] = pd.to_numeric(data[target_column], errors='coerce')
            #     # Drop rows where target coercion failed (e.g. if there were actual strings)
            #     data = data.dropna(subset=[target_column])

            # X = data.drop(columns=[target_column])
            # # Automatically convert any other 'object' columns to numeric if possible
            # # This handles features that were also loaded as strings
            # for col in X.select_dtypes(include=['object']).columns:
            #     X[col] = pd.to_numeric(X[col], errors='ignore')
                
            X_encoded = pd.get_dummies(X, drop_first=True)
            # y = data[target_column]

            # Split data here to ensure we have test labels for the confusion matrix
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import confusion_matrix
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

            # ── AutoGluon ──
            if use_automl:
                automl_config = state.get('automl_config', {})

                # ── Final full training with the config ──
                trained_model, metrics = self._train_with_autogluon(
                    X_train, y_train, problem_type, automl_config
                )
                # y_pred = trained_model.predict(X_test)

                                # Safe prediction handling
                if hasattr(trained_model, 'feature_names_in_'):
                    seen_cols = list(trained_model.feature_names_in_)
                    X_test_pre = X_test.reindex(columns=seen_cols, fill_value=0)
                    y_pred = trained_model.predict(X_test_pre)
                else:
                    y_pred = trained_model.predict(X_test)

            else:
                selected_models = state.get('selected_models', []) or ['RandomForest']
                # trained_model, metrics = self._train_simple_models(X, y, problem_type, selected_models)
                trained_model, metrics = self._train_simple_models(X_train, y_train, state['problem_type'], selected_models)
                # Simple models use scikit-learn predict
                # X_test_simple = pd.get_dummies(X_test, drop_first=True)
                # # Re-align columns in case dummies changed
                # X_test_simple = X_test_simple.reindex(columns=pd.get_dummies(X_train, drop_first=True).columns, fill_value=0)
                # y_pred = trained_model.predict(X_test_simple)

                if hasattr(trained_model, 'feature_names_in_'):
                    seen_cols = list(trained_model.feature_names_in_)
                    X_test_aligned = X_test.reindex(columns=seen_cols, fill_value=0)
                else:
                    X_test_aligned = X_test

                y_pred = trained_model.predict(X_test_aligned)

                # y_pred = trained_model.predict(X_test)
            
            # --- FIX: ADD CONFUSION MATRIX ---
            if problem_type == 'classification':
                # cm = confusion_matrix(y_test, y_pred)
                if len(y_test) > 50_000:
                    sample_idx = y_test.sample(n=50_000, random_state=42).index
                    cm = confusion_matrix(
                        y_test.loc[sample_idx],
                        pd.Series(y_pred, index=y_test.index).loc[sample_idx]
                    )
                else:
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
                # cm_text = f"\n**Confusion Matrix:**\nPredicted 0: [{cm[0][0]}, {cm[0][1]}]\nPredicted 1: [{cm[1][0]}, {cm[1][1]}]"
                if len(cm) > 10:
                            cm_preview = [row[:10] for row in cm[:10]]
                            cm_text = "\n**Confusion Matrix:**\n" + "\n".join([str(row) for row in cm_preview])
                else:
                    cm_text = "\n**Confusion Matrix:**\n" + "\n".join([str(row) for row in cm])

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
            
            rows, features = X.shape
            # --- Large-scale handling ---
            if rows > 1_000_000 or features > 500:
                # Sample 1–5% of data for AutoML to avoid memory issues
                frac = min(0.05, 500_000 / rows)
                X_sample = X.sample(frac=frac, random_state=42)
                y_sample = y.loc[X_sample.index]
                logger.info(f"[AutoGluon] Large dataset detected ({rows} rows, {features} features). Sampling {len(X_sample)} rows for training.")
            else:
                X_sample, y_sample = X, y

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
            # import tempfile
            # predictor_path = os.path.join(tempfile.gettempdir(), f"autogluon_predictor_{os.getpid()}_{int(time.time())}")
            from pathlib import Path
            import time

            # Anchor to dataset directory
            base_dir = Path(self.dataset_path).resolve().parent

            predictor_path = base_dir / "Output" / "AutoGluonModels" / f"run_{int(time.time())}"
            predictor_path.mkdir(parents=True, exist_ok=True)

            predictor_path = str(predictor_path)
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
                'presets': preset,
                # FORCE SEQUENTIAL FOLD FITTING: This is the most important change for 2GB RAM
                'ag_args_ensemble': {
                    'fold_fitting_strategy': 'sequential_local',
                    'use_ray': False
                },
                # DISABLE DYNAMIC STACKING: Prevents the sub-fits that trigger Ray
                'dynamic_stacking': False,
                # REDUCE MODEL COMPLEXITY: Limit to lighter models if needed
                'hyperparameters': {
                    'GBM': {},  # LightGBM is generally memory efficient
                    'XGB': {},
                },
                # SAVE MEMORY: Delete intermediate data after fit
                'save_space': True
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

    def _tune_with_optuna(self, X_train, X_test, y_train, y_test, problem_type: str, model_name: str, n_trials: int = 30):
        """
        Run Optuna HPO for a single model. Returns best model and its score.
        """
        import optuna
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.metrics import accuracy_score, r2_score
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)  # Suppress Optuna noise

        def objective(trial):
            model_name_lower = model_name.lower()
            
            if "randomforest" in model_name_lower:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                }
                model = (RandomForestClassifier(**params, random_state=42, n_jobs=-1) 
                        if problem_type == "classification" 
                        else RandomForestRegressor(**params, random_state=42, n_jobs=-1))

            elif "gradient" in model_name_lower:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
                    "max_depth": trial.suggest_int("max_depth", 2, 10),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                }
                model = (GradientBoostingClassifier(**params, random_state=42) 
                        if problem_type == "classification" 
                        else GradientBoostingRegressor(**params, random_state=42))

            elif "xgb" in model_name_lower or "xgboost" in model_name_lower:
                import xgboost as xgb
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
                    "max_depth": trial.suggest_int("max_depth", 2, 10),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                }
                model = (xgb.XGBClassifier(**params, random_state=42, eval_metric="logloss") 
                        if problem_type == "classification" 
                        else xgb.XGBRegressor(**params, random_state=42))

            elif "logistic" in model_name_lower:
                params = {
                    "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
                    "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
                }
                model = LogisticRegression(**params, max_iter=1000, random_state=42)

            else:
                # Fallback: no tuning, just fit default
                model = (RandomForestClassifier(random_state=42) 
                        if problem_type == "classification" 
                        else RandomForestRegressor(random_state=42))

            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            return accuracy_score(y_test, preds) if problem_type == "classification" else r2_score(y_test, preds)

        direction = "maximize"  # Both accuracy and r2 are maximized
        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Re-train with the best params found
        best_params = study.best_params
        logger.info(f"[Optuna] Best params for {model_name}: {best_params}")
        
        # Rebuild the best model using best_params
        # (reuse the same branching logic for safety)
        model_name_lower = model_name.lower()
        if "randomforest" in model_name_lower:
            best_model = (RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
                        if problem_type == "classification"
                        else RandomForestRegressor(**best_params, random_state=42, n_jobs=-1))
        elif "gradient" in model_name_lower:
            best_model = (GradientBoostingClassifier(**best_params, random_state=42)
                        if problem_type == "classification"
                        else GradientBoostingRegressor(**best_params, random_state=42))
        elif "xgb" in model_name_lower or "xgboost" in model_name_lower:
            import xgboost as xgb
            best_model = (xgb.XGBClassifier(**best_params, random_state=42, eval_metric="logloss")
                        if problem_type == "classification"
                        else xgb.XGBRegressor(**best_params, random_state=42))
        elif "logistic" in model_name_lower:
            best_model = LogisticRegression(**best_params, max_iter=1000, random_state=42)
        else:
            best_model = (RandomForestClassifier(random_state=42)
                        if problem_type == "classification"
                        else RandomForestRegressor(random_state=42))

        best_model.fit(X_train, y_train)
        best_score = study.best_value

        return best_model, best_score, best_params, study
    def _train_simple_models(self, X: pd.DataFrame, y: pd.Series, problem_type: str, model_names: list[str]) -> tuple:
        """
        Train models using simple scikit-learn approach with LLM-selected models.
        Optuna is used for hyperparameter tuning for each model.
        """
        import optuna
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                                    GradientBoostingClassifier, GradientBoostingRegressor)
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.metrics import accuracy_score, r2_score

        optuna.logging.set_verbosity(optuna.logging.WARNING)  # Suppress Optuna noise

        try:
            import xgboost as xgb
            xgb_available = True
        except ImportError:
            xgb_available = False
            logger.warn("XGBoost not available, will use GradientBoosting as alternative")


        if X.shape[0] > 1_000_000 or X.shape[1] > 500:
            frac = min(0.05, 500_000 / X.shape[0])  # max 500k rows
            X = X.sample(frac=frac, random_state=42)
            y = y.loc[X.index]
            logger.info(f"[Simple Models] Large dataset detected. Sampling {len(X)} rows for training.")

        # Ensure we have a valid list to iterate over
        if not model_names or not isinstance(model_names, list):
            model_names = ['RandomForest', 'GradientBoosting'] if problem_type == 'classification' else ['RandomForest']

        logger.info(f"Training simple models with Optuna HPO: {model_names}")

        X_processed = pd.get_dummies(X, drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )

        best_metric_name = 'accuracy' if problem_type == 'classification' else 'r2_score'

        # ─────────────────────────────────────────────
        # Inner helper: build a model from name + params
        # ─────────────────────────────────────────────
        def build_model(model_name: str, params: dict):
            name_lower = model_name.lower()
            if 'randomforest' in name_lower:
                return (RandomForestClassifier(**params, random_state=42, n_jobs=-1)
                        if problem_type == 'classification'
                        else RandomForestRegressor(**params, random_state=42, n_jobs=-1))
            elif ('xgboost' in name_lower or 'xgb' in name_lower) and xgb_available:
                return (xgb.XGBClassifier(**params, random_state=42, eval_metric='logloss')
                        if problem_type == 'classification'
                        else xgb.XGBRegressor(**params, random_state=42))
            elif 'gradient' in name_lower:
                return (GradientBoostingClassifier(**params, random_state=42)
                        if problem_type == 'classification'
                        else GradientBoostingRegressor(**params, random_state=42))
            elif 'logistic' in name_lower:
                return LogisticRegression(**params, max_iter=1000, random_state=42)
            elif 'linear' in name_lower:
                return LinearRegression()
            else:
                # Unknown model name → fallback to RandomForest
                logger.warn(f"Unknown model '{model_name}', falling back to RandomForest")
                return (RandomForestClassifier(random_state=42, n_jobs=-1)
                        if problem_type == 'classification'
                        else RandomForestRegressor(random_state=42, n_jobs=-1))

        # ─────────────────────────────────────────────
        # Inner helper: define the Optuna search space
        # ─────────────────────────────────────────────
        def suggest_params(trial: optuna.Trial, model_name: str) -> dict:
            name_lower = model_name.lower()
            if 'randomforest' in name_lower:
                return {
                    'n_estimators':      trial.suggest_int('n_estimators', 50, 500),
                    'max_depth':         trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features':      trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                }
            elif ('xgboost' in name_lower or 'xgb' in name_lower) and xgb_available:
                return {
                    'n_estimators':      trial.suggest_int('n_estimators', 50, 500),
                    'learning_rate':     trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
                    'max_depth':         trial.suggest_int('max_depth', 2, 10),
                    'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'reg_alpha':         trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                    'reg_lambda':        trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                }
            elif 'gradient' in name_lower:
                return {
                    'n_estimators':  trial.suggest_int('n_estimators', 50, 500),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
                    'max_depth':     trial.suggest_int('max_depth', 2, 10),
                    'subsample':     trial.suggest_float('subsample', 0.5, 1.0),
                    'max_features':  trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                }
            elif 'logistic' in name_lower:
                return {
                    'C':      trial.suggest_float('C', 1e-4, 100.0, log=True),
                    'solver': trial.suggest_categorical('solver', ['lbfgs', 'saga']),
                }
            else:
                # LinearRegression and unknown models have no meaningful hyperparameters to tune
                return {}

        # ─────────────────────────────────────────────
        # Optuna objective factory (one study per model)
        # ─────────────────────────────────────────────
        def make_objective(model_name: str):
            def objective(trial: optuna.Trial) -> float:
                params = suggest_params(trial, model_name)
                model  = build_model(model_name, params)
                model.fit(X_train, y_train)
                preds  = model.predict(X_test)
                return (accuracy_score(y_test, preds)
                        if problem_type == 'classification'
                        else r2_score(y_test, preds))
            return objective

        # ─────────────────────────────────────────────
        # Main training loop — one Optuna study per model
        # ─────────────────────────────────────────────
        best_model      = None
        best_score      = -float('inf')
        best_model_name = None
        all_results     = []

        for model_name in model_names:
            try:
                logger.info(f"[Optuna] Tuning {model_name} ...")

                study = optuna.create_study(
                    direction='maximize',
                    sampler=optuna.samplers.TPESampler(seed=42)
                )
                study.optimize(
                    make_objective(model_name),
                    n_trials=30,              # ← tune this per your time budget
                    show_progress_bar=False
                )

                best_params = study.best_params
                best_trial_score = study.best_value

                # Re-train final model on full training set with the best params found
                final_model = build_model(model_name, best_params)
                final_model.fit(X_train, y_train)

                logger.info(f"[Optuna] {model_name} → best params: {best_params}")
                logger.info(f"[Optuna] {model_name} → {best_metric_name}: {best_trial_score:.4f}")

                all_results.append({
                    'model_name':  model_name,
                    'score':       float(best_trial_score),
                    'best_params': best_params
                })

                if best_trial_score > best_score:
                    best_score      = best_trial_score
                    best_model      = final_model
                    best_model_name = model_name

            except Exception as e:
                logger.warn(f"Failed to tune {model_name}: {str(e)}")
                continue

        # ─────────────────────────────────────────────
        # Fallback if every model in the loop failed
        # ─────────────────────────────────────────────
        if best_model is None:
            logger.warn("All selected models failed — falling back to default RandomForest")
            best_model_name = 'RandomForest'
            best_model = (RandomForestClassifier(random_state=42)
                        if problem_type == 'classification'
                        else RandomForestRegressor(random_state=42))
            best_model.fit(X_train, y_train)
            best_score  = best_model.score(X_test, y_test)
            all_results = [{'model_name': 'RandomForest', 'score': float(best_score), 'best_params': {}}]

        metrics = {
            'best_model':          best_model_name,
            'best_score':          float(best_score),
            'metric_name':         best_metric_name,
            'models_trained':      len(all_results),
            'all_models':          [r['model_name']  for r in all_results],
            'all_scores':          [r['score']        for r in all_results],
            'best_params_per_model': {r['model_name']: r.get('best_params', {}) for r in all_results},
            'training_method':     'Simple+Optuna'
        }

        logger.info(f"Best model: {metrics['best_model']} with {best_metric_name}: {best_score:.4f}")

        return best_model, metrics
    def _save_outputs(self, state: AgentState, output_dir: str = "outputs") -> dict:
        """
        Save all training stage outputs:
        - results.json  → full metrics, configs, reasoning
        - report.md     → human-readable markdown report
        - best_model.pkl → pickled best trained model (sklearn) or note for AutoGluon
        
        Returns:
            dict: paths to all saved files
        """
        import pickle
        from datetime import datetime
        from pathlib import Path

        # ── Setup output directory ──────────────────────────────────────
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        saved_paths = {}

        # ────────────────────────────────────────────────────────────────
        # 1. Build the JSON payload
        # ────────────────────────────────────────────────────────────────
        metrics = state.get('model_metrics', {}) or {}

        # Confusion matrix is a nested list — safe for JSON
        json_payload = {
            "run_timestamp": timestamp,
            "data_path": state.get('data_path'),
            "target_column": state.get('target_column'),
            "problem_type": state.get('problem_type'),

            # ── Model selection decisions ──
            "model_selection": {
                "use_automl": state.get('use_automl'),
                "automl_config": state.get('automl_config'),
                "selected_models": state.get('selected_models'),
                "model_selection_reasoning": state.get('model_selection_reasoning'),
            },

            # ── Training results ──
            "training_results": {
                "training_method": metrics.get('training_method'),
                "best_model": metrics.get('best_model'),
                "best_score": metrics.get('best_score'),
                "metric_name": metrics.get('metric_name', 'score'),
                "models_trained": metrics.get('models_trained'),
                "all_models": metrics.get('all_models', []),
                "all_scores": metrics.get('all_scores', []),
                "confusion_matrix": metrics.get('confusion_matrix'),        # list of lists or None
                "best_params_per_model": metrics.get('best_params_per_model', {}),
                "optuna_refined_config": metrics.get('optuna_refined_config'),  # None if not used
            },

            # ── Agent conversation history ──
            "agent_messages": state.get('agent_messages', []),

            # ── Workflow metadata ──
            "workflow": {
                "final_step": state.get('step'),
                "error": state.get('error'),
            }
        }

        json_path = out_dir / f"results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_payload, f, indent=2, default=str)   # default=str handles numpy types
        saved_paths['json'] = str(json_path)
        logger.info(f"[Save Outputs] JSON saved → {json_path}")

        # ────────────────────────────────────────────────────────────────
        # 2. Build the Markdown report
        # ────────────────────────────────────────────────────────────────
        use_automl   = state.get('use_automl', False)
        best_score   = metrics.get('best_score', 0) or 0
        best_model   = metrics.get('best_model', 'N/A')
        problem_type = state.get('problem_type', 'N/A')

        # Build model comparison table rows
        all_models = metrics.get('all_models', [])
        all_scores = metrics.get('all_scores', [])
        model_rows = ""
        for name, score in zip(all_models, all_scores):
            marker = " ✅" if name == best_model else ""
            model_rows += f"| {name}{marker} | {float(score):.4f} |\n"

        # Build confusion matrix block (classification only)
        cm_block = ""
        cm = metrics.get('confusion_matrix')
        if cm:
            cm_block = f"""
    ## Confusion Matrix

    |  | Predicted 0 | Predicted 1 |
    |---|---|---|
    | **Actual 0** | {cm[0][0]} | {cm[0][1]} |
    | **Actual 1** | {cm[1][0]} | {cm[1][1]} |

    - **True Negatives (TN):** {cm[0][0]}
    - **False Positives (FP):** {cm[0][1]}
    - **False Negatives (FN):** {cm[1][0]}
    - **True Positives (TP):** {cm[1][1]}
    """

        # Build Optuna config block (if used)
        optuna_block = ""
        refined_cfg = metrics.get('optuna_refined_config')
        if refined_cfg:
            optuna_block = f"""
    ## Optuna-Refined AutoGluon Config

    | Parameter | Value |
    |---|---|
    | Models | {refined_cfg.get('models')} |
    | Time Limit | {refined_cfg.get('time_limit')}s |
    | Preset | {refined_cfg.get('preset')} |
    """

        # Build per-model best params block (Simple+Optuna only)
        params_block = ""
        best_params_per_model = metrics.get('best_params_per_model', {})
        if best_params_per_model:
            params_block = "## Best Hyperparameters per Model\n\n"
            for model_name, params in best_params_per_model.items():
                params_block += f"### {model_name}\n```json\n{json.dumps(params, indent=2)}\n```\n\n"

        # Build agent messages block
        messages_block = ""
        agent_msgs = state.get('agent_messages', [])
        if agent_msgs:
            messages_block = "## Agent Reasoning Log\n\n"
            for msg in agent_msgs:
                agent_name = msg.get('agent', 'unknown').replace('_', ' ').title()
                messages_block += f"### {agent_name} Agent\n{msg.get('message', '')}\n\n---\n\n"

        md_report = f"""# AutoML Agent — Run Report
    **Generated:** {timestamp}  
    **Data Path:** `{state.get('data_path')}`  
    **Target Column:** `{state.get('target_column')}`  
    **Problem Type:** `{problem_type}`  

    ---

    ## Model Selection Decision

    | Field | Value |
    |---|---|
    | Approach | {'AutoGluon (AutoML)' if use_automl else 'Simple Training + Optuna'} |
    | Best Model | {best_model} |
    | Best Score | {best_score:.4f} |
    | Models Trained | {metrics.get('models_trained', 0)} |

    ### LLM Selection Reasoning
    {state.get('model_selection_reasoning', '_Not available_')}

    ---

    ## Model Comparison

    | Model | Score |
    |---|---|
    {model_rows}
    {cm_block}
    {optuna_block}
    {params_block}
    ---

    {messages_block}
    ## Workflow Status

    | Field | Value |
    |---|---|
    | Final Step | `{state.get('step')}` |
    | Error | `{state.get('error') or 'None'}` |
    """

        md_path = out_dir / f"report_{timestamp}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_report)
        saved_paths['markdown'] = str(md_path)
        logger.info(f"[Save Outputs] Markdown saved → {md_path}")

        # ────────────────────────────────────────────────────────────────
        # 3. Save the best model as pickle
        # ────────────────────────────────────────────────────────────────
        trained_model = state.get('trained_model')
        if trained_model is not None:
            pkl_path = out_dir / f"best_model_{timestamp}.pkl"

            if use_automl:
                # AutoGluon predictors have their own save system (they write a directory).
                # We save a lightweight pickle that stores the predictor's path so it
                # can be reloaded anywhere with TabularPredictor.load(path).
                ag_path = getattr(trained_model, 'path', None)
                pickle_payload = {
                    "model_type": "AutoGluon",
                    "autogluon_predictor_path": ag_path,
                    "best_model_name": best_model,
                    "best_score": best_score,
                    "reload_instructions": (
                        "from autogluon.tabular import TabularPredictor; "
                        f"predictor = TabularPredictor.load('{ag_path}')"
                    )
                }
                with open(pkl_path, 'wb') as f:
                    pickle.dump(pickle_payload, f)
                logger.info(f"[Save Outputs] AutoGluon predictor path pickled → {pkl_path}")
                logger.info(f"[Save Outputs] To reload: TabularPredictor.load('{ag_path}')")
            else:
                # Sklearn / XGBoost models pickle cleanly
                with open(pkl_path, 'wb') as f:
                    pickle.dump(trained_model, f)
                logger.info(f"[Save Outputs] Sklearn model pickled → {pkl_path}")

            saved_paths['pickle'] = str(pkl_path)
        else:
            logger.warn("[Save Outputs] No trained model found in state — skipping pickle.")

        # ────────────────────────────────────────────────────────────────
        # 4. Summary log
        # ────────────────────────────────────────────────────────────────
        logger.info("[Save Outputs] ── Saved files ──────────────────────")
        for file_type, path in saved_paths.items():
            logger.info(f"[Save Outputs]   {file_type:10s} → {path}")
        logger.info("[Save Outputs] ─────────────────────────────────────")

        return saved_paths
    
    # AFTER
    def run(self, data_path: str, target_column: str = None, output_dir: str = "outputs", automl_directives: dict = None, problem_type: str = None) -> dict:
        """
        Run the complete AutoML workflow.

        Args:
            data_path:     Path to the data file
            target_column: Optional target column name (auto-detected if not provided)
            output_dir:    Directory to save JSON, Markdown, and pickle outputs

        Returns:
            dict: Final state with results + 'saved_files' key
        """
        self.dataset_path = data_path
        self.target_column = target_column
        self.output_dir = output_dir
        initial_state = {
        'data_path': data_path,
        'target_column': target_column,
        'data': None,
        'data_summary': None,
        'problem_type': problem_type,           # ← was None hardcoded, now uses argument
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
        'agent_messages': [],
        'automl_directives': automl_directives, # ← NEW
        'human_approved': None,
    }
        logger.info("Starting AutoML agent workflow")
        final_state = self.graph.invoke(initial_state)

        if final_state.get('error'):
            logger.error(f"Workflow completed with error: {final_state['error']}")
        else:
            logger.info("Workflow completed successfully!")

        # ── NEW: Save all outputs ────────────────────────────────────
        try:
            saved_files = self._save_outputs(final_state, output_dir=output_dir)
            final_state['saved_files'] = saved_files
        except Exception as e:
            logger.error(f"[Save Outputs] Failed to save outputs: {str(e)}", e)
            final_state['saved_files'] = {}
        # ─────────────────────────────────────────────────────────────

        return final_state



