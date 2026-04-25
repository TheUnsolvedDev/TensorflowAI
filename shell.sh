
# server to home
rsync -avzh myserver:/mnt/storage/da24d402/Documents/TensorflowAI/ ./TensorflowAI/

# home to server
rsync -avzh ./TensorflowAI/ myserver:/mnt/storage/da24d402/Documents/TensorflowAI/

