# BRATS DATASET 2D PRE-PROCESSOR FOR YOLOV11-SEG

## What is it for? 
If you're late to the BraTS competition but still want to train your model, validate it and test it, and you're working with a 2D YOLOv11-seg model, this is the repository for you

## How to use
1. Download your own copy of the BraTS Dataset, and rename the training dataset to "raw_files" and put it in the main directory. We will only use the training dataset
2. Give permission to both process_all_mod.sh and process_stack.sh, by using: chmod +x [name_of_script].sh
3. Run ./process_all_mod to create dataset split for all 4 modality. Do not run the scripts separately. This will create 4 directories, with images and labels split, and test, train and val splits. Note that the sh script will automatically remove unnecesary/intermediate diretories for you. 
4. Run ./process_stack.sh if you want to create a separate dataset that stacks 3 modality to create a single PNG image. By default, it uses ["t1c", "t2f", "t2w"], you can change it in the hyperparameter section of the stack_images.py