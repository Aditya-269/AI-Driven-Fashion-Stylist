import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from PIL import Image
import os

# Define dataset folder and persistent Chroma client
dataset_folder = 'Data'
chroma_client = chromadb.PersistentClient(path="Vector_database")
image_loader = ImageLoader()
CLIP = OpenCLIPEmbeddingFunction()

# Create or get the 'image' collection in Chroma
image_vdb = chroma_client.get_or_create_collection(name="image", embedding_function=CLIP, data_loader=image_loader)

ids = []
uris = []
embeddings = []

# Loop through images in the dataset folder
for i, filename in enumerate(sorted(os.listdir(dataset_folder))):
    if filename.endswith('.png'):
        file_path = os.path.join(dataset_folder, filename)
        
        # Append the image id and URI
        ids.append(str(i))
        uris.append(file_path)
        
        # Open the image and compute its embedding
        image = Image.open(file_path)
        embedding = CLIP.embed(image)  # Compute embedding for the image
        embeddings.append(embedding)

# Add the image embeddings and metadata to the Chroma database
image_vdb.add(
    ids=ids,
    uris=uris,
    embeddings=embeddings
)

print("Images stored to the Vector database.")
