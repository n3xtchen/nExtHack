
PROJ_NAME='demo'
PROJ_VER='0.0.1'
AUTHOR='n3xtchen'
EMAIL='echenwen@gmail.com'
PYTHON_VER='3.8'
LICENSE='MIT'
HOMEPAGE="https://github.com/pypa/sampleproject"
ISSUES="https://github.com/pypa/sampleproject/issues"

# echo > pyproject.toml << EOF
cat > pyproject.toml << EOF
[project]
name = "${PROJ_NAME}"
version = "${PROJ_VER}"
authors = [
  { name="${AUTHOR}", email="${EMAIL}" },
]
description = "自动文章发送工具"
readme = "README.md"
requires-python = "${PYTHON_VER}"
classifiers = [
    "Programming Language :: Python :: 3",
    "Operating System :: OS Independent",
]
license = "${LICENSE}"
license-files = ["LICEN[CS]E*"]

[project.urls]
Homepage = "${HOMEPAGE}"
Issues = "${ISSUES}"

[build-system]
requires = ["setuptools >= 77.0.3"]
build-backend = "setuptools.build_meta"
EOF
