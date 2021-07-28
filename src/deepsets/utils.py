from datetime import datetime
from hashlib import md5
import json
from pathlib import Path


def encode_dict(dict_:dict):
    return md5(repr(dict_).encode("utf-8")).hexdigest()

def outdir_for_run(config:dict):
    config_path = Path('.runs') / encode_dict(config)
    run_path = config_path / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_path.mkdir(parents=True, exist_ok=True)

    config_file_path = config_path / 'config.json'
    config_json = json.dumps(config, sort_keys=True, indent=4
    )
    config_file_path.write_text(config_json)
    return run_path