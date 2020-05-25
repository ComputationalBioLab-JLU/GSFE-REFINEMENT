# GSFE-REFINEMENT
protein refinement by generalized solvation free energy theory 

# Title

![banner]()

![badge]()
![badge]()
[![license](https://img.shields.io/github/license/:user/:repo.svg)](LICENSE)
[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

This is an example file with maximal choices selected.

This is a long description.

## Table of Contents

- [Security](#security)
- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [API](#api)
- [Contributing](#contributing)
- [License](#license)

## Security

### Any optional sections

## Background

### Any optional sections

## Install

This module depends upon a knowledge of [Markdown]().

```
```

### Any optional sections

## Usage

```
```

Note: The `license` badge image link at the top of this file should be updated with the correct `:user` and `:repo`.

### Any optional sections

## API

### Any optional sections

## More optional sections

## Contributing

See [the contributing file](CONTRIBUTING.md)!

PRs accepted.

Small note: If editing the Readme, please conform to the [standard-readme](https://github.com/RichardLitt/standard-readme) specification.

### Any optional sections

## License

[MIT © Richard McRichface.](../LICENSE)



需要安装的软件和版本，python3,  biopython (1.74)，numpy (1.17.2)，torch (1.2.0)，

另外需要安装Java(openjdk version "1.8.0_191")



如何跑程序："python3 auto_opi_mutation_model770_sincos.py --PATH native_start --native_name 101m__native.pdb --decoy_name 101m__model.pdb --device cpu --L1_smooth_parameter 1.2 --ENRAOPY_W 1"

命令行可选参数参数: --PATH 需要优化的start model 和 native structure 的路径

		  --native_name native structure 文件名字，

 		--decoy_name 需要refinement的decoy文件， 

		--device 使用cpu或者是gpu，

		--L1_smooth_parameter loss——sml1参数，

		--entropy_w 是否使用熵权重
