import subprocess
print(subprocess.getoutput('nvidia-smi'))
subprocess.check_call(['pip','-q','install','transformers==4.57.1','accelerate','huggingface_hub'])
