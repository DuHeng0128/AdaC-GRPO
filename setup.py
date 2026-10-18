# Copyright 2025 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# See LICENSES/Apache-2.0.txt and THIRD_PARTY_NOTICES.md.
# AdaC-GRPO release metadata and packaging: copyright 2026 Sida Li.

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent

setup(
    name="adac-grpo",
    version="0.1.0",
    author="Sida Li and collaborators",
    description="Adaptive curriculum GRPO for visual Sudoku (Findings of EMNLP 2026)",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/DuHeng0128/AdaC-GRPO",
    project_urls={"Paper": "https://arxiv.org/abs/2610.13184"},
    license="MIT; inherited components under Apache-2.0",
    license_files=["LICENSE", "LICENSES/Apache-2.0.txt", "THIRD_PARTY_NOTICES.md"],
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.10",
    install_requires=[],
    extras_require={
        "data": ["pillow>=10"],
        "train": [
            "torch>=2.5.1,<2.7",
            "transformers>=4.49,<4.52",
            "trl==0.16.0",
            "accelerate>=1.2.1,<2",
            "datasets>=3.2,<4",
            "deepspeed==0.15.4",
            "qwen-vl-utils>=0.0.8",
            "einops>=0.8",
            "pillow",
            "numpy",
            "nltk",
            "rouge-score",
            "wandb>=0.19.1",
            "liger-kernel==0.5.2",
            "peft>=0.14",
        ],
        "vllm": ["vllm==0.7.2"],
    },
)
