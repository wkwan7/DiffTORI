import os
import re
from omegaconf import OmegaConf


def parse_cfg(cfg_path: str, vv=None) -> OmegaConf:
	"""Parses a config file and returns an OmegaConf object."""
	base = OmegaConf.load(cfg_path / 'default.yaml')
	if vv is None:
		cli = OmegaConf.from_cli()
	else:
		cli = vv.dump()
	for k,v in cli.items():
		if v == None:
			cli[k] = True
	base.merge_with(cli)

	# Modality config
	if cli.get('modality', base.modality) not in {'state', 'pixels'}:
		raise ValueError('Invalid modality: {}'.format(cli.get('modality', base.modality)))
	modality = cli.get('modality', base.modality)
	if modality != 'state':
		mode = OmegaConf.load(cfg_path / f'{modality}.yaml')
		base.merge_with(mode, cli)

	# Task config
	try:
		domain, task = base.task.split('-', 1)
	except:
		raise ValueError(f'Invalid task name: {base.task}')
	task_path = cfg_path / 'tasks' / f'{base.task}.yaml'
	domain_path = cfg_path / 'tasks' / f'{domain}.yaml'
	if os.path.exists(task_path):
		domain_path = task_path
	if not os.path.exists(domain_path):
		domain_path = cfg_path / 'tasks' / 'default.yaml'
	domain_cfg = OmegaConf.load(domain_path)
	base.merge_with(domain_cfg, cli)

	# Algebraic expressions
	for k,v in base.items():
		if isinstance(v, str):
			match = re.match(r'(\d+)([+\-*/])(\d+)', v)
			if match:
				base[k] = eval(match.group(1) + match.group(2) + match.group(3))
				if isinstance(base[k], float) and base[k].is_integer():
					base[k] = int(base[k])

	# Convenience
	base.task_title = base.task.replace('-', ' ').title()
	#base.device = 'cuda' if base.modality == 'state' else 'cpu'
	base.exp_name = str(base.get('exp_name', 'default'))

	### to make sure vv value can overwrite all existing values
	if vv is not None:
		base.merge_with(vv.dump())

	return base
