import sys
import json
from pathlib import Path

# ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.query import Retriever, synthesize_answer


r = Retriever('data/faiss.index')
ret = r.query('What is the standard delivery timeline?', top_k=5)
print(json.dumps(ret, ensure_ascii=False, indent=2))
out = synthesize_answer(ret)
print('\nSYNTHESIS:\n')
print(json.dumps(out, ensure_ascii=False, indent=2))
