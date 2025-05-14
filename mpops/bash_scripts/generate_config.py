from omegaconf import OmegaConf
import os
import sys
sys.path.append('..')

params_source = sys.argv[1]
params_dir = sys.argv[2]

conf = OmegaConf.load(params_source)
print('Loaded configuration:', conf)

degrees = conf.random_search[1].get('degree', [])
if not degrees:
    print('Error: No degrees found in config!')
    exit(1)

os.makedirs(params_dir, exist_ok=True)

for degree in degrees:
    modified_conf = OmegaConf.load(params_source)
    modified_conf.random_search[1]['degree'] = [int(degree)]

    config_file = os.path.join(params_dir, f'params_degree{degree}.yml')
    with open(config_file, 'w') as f:
        OmegaConf.save(config=modified_conf, f=f)

    print(f'Generated config for degree {degree}')
