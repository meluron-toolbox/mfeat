#!/usr/bin/env python3

#---------------------------------
# Author: Ankit Anand
# Date: 10/08/25
# Email: ankit0.anand0@gmail.com
#---------------------------------

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]/"src"/"mfeat"))

print(sys.path)
from loudness import get_loudness
