需要安装的软件和版本，python3,  biopython (1.74)，numpy (1.17.2)，torch (1.2.0)，
另外需要安装Java(openjdk version "1.8.0_191")

如何跑程序："python3 auto_opi_mutation_model770_sincos.py --PATH native_start --native_name 101m__native.pdb --decoy_name 101m__model.pdb --device cpu --L1_smooth_parameter 1.2 --ENRAOPY_W 1"
命令行可选参数参数: --PATH 需要优化的start model 和 native structure 的路径
		  --native_name native structure 文件名字，
 		--decoy_name 需要refinement的decoy文件， 
		--device 使用cpu或者是gpu，
		--L1_smooth_parameter loss——sml1参数，
		--entropy_w 是否使用熵权重
