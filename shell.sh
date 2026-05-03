
# server to home
rsync -avzh --exclude='.git' myserver:/mnt/storage/da24d402/Documents/TensorflowAI/ /home/shuvrajeet/Documents/TensorflowAI/
rsync -avzh --exclude='.git' myserver:/mnt/storage/da24d402/Documents/TensorflowAI/ComputerVision/GenerativeAdvesarialNetworks/ /home/shuvrajeet/Documents/TensorflowAI/ComputerVision/GenerativeAdvesarialNetworks/

# home to server
rsync -avh --exclude='.git' /home/shuvrajeet/Documents/TensorflowAI/ myserver:/mnt/storage/da24d402/Documents/TensorflowAI/
rsync -avh --exclude='.git' /home/shuvrajeet/Documents/TensorflowAI/ComputerVision/GenerativeAdvesarialNetworks/ myserver:/mnt/storage/da24d402/Documents/TensorflowAI/ComputerVision/GenerativeAdvesarialNetworks/
