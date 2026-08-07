from src.models.model_runner import ModelRunner
from src.hooks.hook_manager import HookManager


runner = ModelRunner()

runner.load_model()

hook_manager = HookManager(runner.model)

hook_manager.register_hooks()

outputs = runner.infer(
    "Hello world!"
)

hook_manager.remove_hooks()

print(type(outputs))
print(outputs.keys())
print(outputs.last_hidden_state.shape)

# for name, module in runner.model.named_modules():
#     print(name)

print("hooks info: \n")
records = hook_manager.get_records()
for record in records:
    print(record)
