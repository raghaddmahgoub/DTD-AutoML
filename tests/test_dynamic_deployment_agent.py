"""
Test script for the dynamic DeploymentAgent.
"""
import os
import sys
import shutil
import pickle
import logging
from pathlib import Path
from dotenv import load_dotenv

# Setup project root import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from agents.dynamic.deployment_agent.deployment_agent import DeploymentAgent
from state.pipeline_state import make_initial_state
from src.utils.logger import Logger

def test_deployment_agent_basic():
    print("\n" + "=" * 70)
    print("DEPLOYMENT AGENT — ISOLATED UNIT TEST")
    print("=" * 70 + "\n")

    # 1. Setup temporary directories and files
    temp_dir = PROJECT_ROOT / "Output" / "test_temp_deployment"
    temp_dir.mkdir(parents=True, exist_ok=True)

    dummy_model_path = temp_dir / "test_model.pkl"
    dummy_split_path = temp_dir / "X_train.csv"

    # Train a tiny scikit-learn model
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd

    print("-> Creating mock classification dataset and model...")
    X, y = make_classification(n_samples=20, n_features=4, random_state=42)
    model = RandomForestClassifier(n_estimators=3, random_state=42)
    model.fit(X, y)
    
    # Store feature names to check Pydantic and FastAPI integration
    model.feature_names_in_ = ["age", "fare", "class_type", "embarked"]
    
    with open(dummy_model_path, "wb") as f:
        pickle.dump(model, f)
        
    df_train = pd.DataFrame(X, columns=model.feature_names_in_)
    df_train.to_csv(dummy_split_path, index=False)

    # 2. Setup pipeline state
    run_id = "test-deploy-isolated-run"
    state = make_initial_state(
        data_path="uploads/Titanic-Dataset.csv",
        nl_query="deploy my classifier to production",
        run_id=run_id
    )
    
    state["trained_model_path"] = str(dummy_model_path)
    state["X_train_path"] = str(dummy_split_path)
    state["task_type"] = "classification"
    state["model_metrics"] = {
        "best_model": "RandomForestClassifier",
        "best_score": 0.85,
        "training_method": "train_simple"
    }

    # 3. Instantiate and run DeploymentAgent
    logger_obj = Logger()
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.0,
    )
    agent = DeploymentAgent(logger_obj=logger_obj, llm=llm)
    
    print("-> Running DeploymentAgent...")
    updated_state = agent.run(state)
    
    # 4. Assert and verify outputs
    status = updated_state.get("status")
    print(f"\n[OK] Runner Status: {status}")
    print(f"[OK] Agent Step: {updated_state.get('step')}")
    
    if status == "success":
        package_path = updated_state.get("deployment_package_path")
        endpoint_url = updated_state.get("endpoint_url")
        agent_out = updated_state.get("agent_outputs", {}).get("deployment", {})

        print(f"\n[DATA] Deployment outputs:")
        print(f"   * package_path : {package_path}")
        print(f"   * endpoint_url : {endpoint_url}")
        
        # Verify files are written to folder
        pkg_dir = Path(package_path)
        assert pkg_dir.exists(), "Deployment directory was not created!"
        assert (pkg_dir / "api_server.py").exists(), "api_server.py is missing!"
        assert (pkg_dir / "Dockerfile").exists(), "Dockerfile is missing!"
        assert (pkg_dir / "requirements.txt").exists(), "requirements.txt is missing!"
        assert (pkg_dir / "model.pkl").exists(), "model.pkl was not copied!"
        
        print("\n[OK] All deployment files verified successfully on disk!")
        
        # Output snippet of generated API code
        with open(pkg_dir / "api_server.py", "r", encoding="utf-8") as f:
            code = f.read()
        print("\n--- GENERATED API SERVER SNIPPET ---")
        print("\n".join(code.splitlines()[:25]))
        print("------------------------------------")
    else:
        print(f"[ERROR] Error encountered: {updated_state.get('error')}")

    # 5. Cleanup temp directories
    print("\n-> Cleaning up temporary files...")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        
    print("\n" + "=" * 70 + "\n")


def test_deployment_agent_with_feedback():
    print("\n" + "=" * 70)
    print("DEPLOYMENT AGENT — FEEDBACK TEST")
    print("=" * 70 + "\n")

    # 1. Setup temporary directories and files
    temp_dir = PROJECT_ROOT / "Output" / "test_temp_deployment_fb"
    temp_dir.mkdir(parents=True, exist_ok=True)

    dummy_model_path = temp_dir / "test_model_fb.pkl"
    dummy_split_path = temp_dir / "X_train_fb.csv"

    # Train a tiny scikit-learn model
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd

    print("-> Creating mock classification dataset and model...")
    X, y = make_classification(n_samples=20, n_features=4, random_state=42)
    model = RandomForestClassifier(n_estimators=3, random_state=42)
    model.fit(X, y)
    model.feature_names_in_ = ["age", "fare", "class_type", "embarked"]
    
    with open(dummy_model_path, "wb") as f:
        pickle.dump(model, f)
        
    df_train = pd.DataFrame(X, columns=model.feature_names_in_)
    df_train.to_csv(dummy_split_path, index=False)

    # 2. Setup pipeline state WITH human feedback
    run_id = "test-deploy-fb-run"
    state = make_initial_state(
        data_path="uploads/Titanic-Dataset.csv",
        nl_query="deploy my classifier to production",
        run_id=run_id
    )
    
    state["trained_model_path"] = str(dummy_model_path)
    state["X_train_path"] = str(dummy_split_path)
    state["task_type"] = "classification"
    state["model_metrics"] = {
        "best_model": "RandomForestClassifier",
        "best_score": 0.85,
        "training_method": "train_simple"
    }
    
    # Inject user feedback simulating human feedback cycle
    state["feedback_history"] = [{
        "agent": "deployment",
        "feedback_text": "Please add a '/metadata' GET endpoint to the api_server.py code and include lightgbm in the requirements.txt file.",
        "iteration": 1
    }]

    # 3. Instantiate and run DeploymentAgent with gemini-2.5-flash-lite
    logger_obj = Logger()
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.0,
    )
    agent = DeploymentAgent(logger_obj=logger_obj, llm=llm)
    
    print("-> Running DeploymentAgent with feedback context...")
    updated_state = agent.run(state)
    
    # 4. Assert and verify outputs
    status = updated_state.get("status")
    print(f"\n[OK] Runner Status: {status}")
    print(f"[OK] Agent Step: {updated_state.get('step')}")
    
    if status == "success":
        package_path = updated_state.get("deployment_package_path")
        pkg_dir = Path(package_path)
        
        # Verify files are written to folder
        assert pkg_dir.exists(), "Deployment directory was not created!"
        assert (pkg_dir / "api_server.py").exists(), "api_server.py is missing!"
        assert (pkg_dir / "requirements.txt").exists(), "requirements.txt is missing!"
        
        with open(pkg_dir / "api_server.py", "r", encoding="utf-8") as f:
            api_code = f.read()
        with open(pkg_dir / "requirements.txt", "r", encoding="utf-8") as f:
            reqs = f.read()

        print("\n[VERIFY] Checking if feedback was applied:")
        has_metadata = "/metadata" in api_code
        has_lightgbm = "lightgbm" in reqs.lower()

        print(f"   * Has '/metadata' endpoint in api_server.py : {has_metadata}")
        print(f"   * Has 'lightgbm' in requirements.txt       : {has_lightgbm}")

        assert has_metadata, "Feedback not applied: '/metadata' route is missing from generated app server!"
        assert has_lightgbm, "Feedback not applied: 'lightgbm' is missing from requirements.txt!"
        print("\n[OK] Human-in-the-loop feedback successfully processed and applied by the agent!")
    else:
        print(f"[ERROR] Error encountered: {updated_state.get('error')}")

    # 5. Cleanup temp directories
    print("\n-> Cleaning up temporary files...")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    test_deployment_agent_basic()
    test_deployment_agent_with_feedback()
