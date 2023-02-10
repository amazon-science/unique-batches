# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from datetime import datetime, timezone
from pathlib import Path

from unique_batches.utils.framework import Stage

DEFAULT_LOG_DIR = "logs"
DEFAULT_EXPERIMENT_NAME = datetime.now(tz=timezone.utc).isoformat()

DEFAULT_SPLIT_RATIO = {
    Stage.train: 0.8,
    Stage.val: 0.1,
    Stage.test: 0.1,
}
RARE_DOMAIN_SIZE = 10

ANNOTATION = "annotation"
TAGS = "tags"
UTTERANCE_TEXT = "utterance_text"
DOMAIN = "domain"
INTENT = "intent"


PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
CONFIG_PATH = str(PROJECT_ROOT / "configs")
CONFIG_NAME = "main.yaml"
