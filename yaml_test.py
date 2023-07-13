import yaml

with open('hparams.yml', 'r') as file:
    prime_service = yaml.safe_load(file)

print(prime_service)