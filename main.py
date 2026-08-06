from src.models.model_runner import ModelRunner


runner = ModelRunner()

runner.load_model()

outputs = runner.infer(
    "Hello world!"
)

print(type(outputs))
print(outputs.keys())
print(outputs.last_hidden_state.shape)

for name, module in runner.model.named_modules():
    print(name)