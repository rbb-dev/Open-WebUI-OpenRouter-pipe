from setuptools import setup, find_packages

setup(
    name="openrouter_modules",
    version="2.0.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "aiohttp>=3.9.0",
        "cryptography>=41.0.0",
        "fastapi>=0.100.0",
        "httpx>=0.25.0",
        "lz4>=4.0.0",
        "pydantic>=2.0.0",
        "pydantic_core>=2.0.0",
        "sqlalchemy>=2.0.0",
        "tenacity>=8.0.0"
    ],
    author="rbb-dev",
    author_email="",
    description="Modular components for OpenRouter Responses API pipe (Open WebUI)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rbb-dev/openrouter_responses_pipe",
    project_urls={
        "Bug Tracker": "https://github.com/rbb-dev/openrouter_responses_pipe/issues",
        "Documentation": "https://github.com/rbb-dev/openrouter_responses_pipe/blob/main/docs/documentation_index.md",
        "Source": "https://github.com/rbb-dev/openrouter_responses_pipe",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    license="MIT",
)
