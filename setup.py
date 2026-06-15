from setuptools import setup, find_packages

setup(
    name="opengoalrl",
    version="0.2.0",
    description="Scenario-based reinforcement learning toolkit for Google Research Football",
    author="OpenGoalRL Contributors",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "gfootball>=2.10",
        "gymnasium>=0.29",
        "stable-baselines3>=2.1",
        "pyyaml>=6.0",
        "numpy>=1.23,<2.0",
        "torch>=2.0",
        "matplotlib>=3.7",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
        "test": [
            "pytest>=7.0",
            "stable-baselines3>=2.1",
            "torch>=2.0",
            "gymnasium>=0.29",
            "numpy>=1.23,<2.0",
            "pyyaml>=6.0",
            "matplotlib>=3.7",
        ],
        "docs": ["mkdocs-material>=9.5", "mkdocs"],
    },
    entry_points={
        "console_scripts": [
            "opengoalrl-train=opengoalrl.scripts.train:main",
            "opengoalrl-eval=opengoalrl.scripts.evaluate:main",
            "opengoalrl-baseline=opengoalrl.scripts.baseline:main",
            "opengoalrl-curriculum=opengoalrl.scripts.curriculum_train:main",
            "opengoalrl-generate-scenario=opengoalrl.scripts.generate_scenario:main",
            "opengoalrl-diagnose=opengoalrl.scripts.diagnose:main",
            "opengoalrl-report=opengoalrl.scripts.report:main",
            "opengoalrl-auto-curriculum=opengoalrl.scripts.auto_curriculum:main",
            "opengoalrl-coach=opengoalrl.scripts.coach:main",
            "opengoalrl-train-ma=opengoalrl.scripts.train_ma:main",
            "opengoalrl-download-models=opengoalrl.scripts.download_models:main",
        ],
    },
)
