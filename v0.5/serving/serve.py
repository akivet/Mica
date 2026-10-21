"""Start the judgment server (POST /v1/systemone). Run: python serve.py --help"""
import importlib.machinery, importlib.util, json, os, sys
from pathlib import Path

LIB = Path(os.path.dirname(os.path.abspath(__file__))) / 'lib'


def load():
    sys.path.insert(0, str(LIB))
    name = json.loads((LIB / 'index.json').read_text(encoding='utf-8'))['runtime']
    # lib/ may hold native extension modules (one ABI-tagged file per Python version and platform) or .pyc; the
    # finder picks the file that matches the running interpreter. .py is for the triton.jit kernel module only.
    finder = importlib.machinery.FileFinder(str(LIB),
        (importlib.machinery.ExtensionFileLoader, list(importlib.machinery.EXTENSION_SUFFIXES)),
        (importlib.machinery.SourcelessFileLoader, ['.pyc']),
        (importlib.machinery.SourceFileLoader, ['.py']))
    spec = finder.find_spec(name)
    if spec is None:
        raise SystemExit('this runtime has no module for Python %d.%d on this platform' % sys.version_info[:2])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == '__main__':
    load().main(sys.argv[1:])
