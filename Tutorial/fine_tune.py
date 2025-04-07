import subprocess
import re
import ast
import astor
import optuna

TUTORIAL_FILE = "/home/naba/Desktop/IMC-Prosperity-3/Tutorial/tutorial_sub4.py"

def modify_params(params):
    with open(TUTORIAL_FILE, "r") as f:
        source = f.read()

    tree = ast.parse(source)

    class ParamUpdater(ast.NodeTransformer):
        def visit_Assign(self, node):
            if (isinstance(node.targets[0], ast.Name) and
                node.targets[0].id == "PARAMS"):
                
                # Create new PARAMS dict
                new_dict = {
                    "KELP": {
                        "take_width": params["kelp_take"],
                        "clear_width": 0,
                        "prevent_adverse": True,
                        "adverse_volume": 10,
                        "reversion_beta": params["kelp_beta"],
                        "disregard_edge": 1,
                        "join_edge": 0,
                        "default_edge": 1,
                        "soft_position_limit": params["kelp_pos"],
                    },
                    "RAINFOREST_RESIN": {
                        "fair_value": 10000,
                        "take_width": params["resin_take"],
                        "clear_width": 0,
                        "disregard_edge": 1,
                        "join_edge": 2,
                        "default_edge": 4,
                        "soft_position_limit": params["resin_pos"],
                    },
                }

                node.value = ast.parse(repr(new_dict)).body[0].value
                return node
            return node

    updated_tree = ParamUpdater().visit(tree)

    # Save updated code
    with open(TUTORIAL_FILE, "w") as f:
        f.write(astor.to_source(updated_tree))

def run_backtest():
    cmd = f"prosperity3bt {TUTORIAL_FILE} 0"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    output = result.stdout

    kelp_match = re.search(r"KELP:\s*(\d+)", output)
    resin_match = re.search(r"RAINFOREST_RESIN:\s*(\d+)", output)
    total_match = re.search(r"Total profit:\s*(\d+)", output)

    if kelp_match and resin_match and total_match:
        kelp = int(kelp_match.group(1))
        resin = int(resin_match.group(1))
        total = int(total_match.group(1))
        return kelp, resin, total
    return 0, 0, 0

def objective(trial):
    params = {
        "kelp_take": trial.suggest_int("kelp_take", 1, 3),
        "kelp_beta": trial.suggest_float("kelp_beta", -0.5, 0.0),
        "kelp_pos": trial.suggest_int("kelp_pos", 5, 20),
        "resin_take": trial.suggest_int("resin_take", 1, 5),
        "resin_pos": trial.suggest_int("resin_pos", 5, 15),
    }

    modify_params(params)
    kelp, resin, total = run_backtest()

    # Constraint: each individual product must beat current best
    if kelp > 0 and resin > 0:
        return total
    return 0  # Penalize if not met

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best parameters:", study.best_params)
