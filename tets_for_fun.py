import numpy as np
import boto3
from io import BytesIO
from PIL import Image

# Convert the numpy array to a PIL Image object
def aws_textract(your_numpy_array):
  img = Image.fromarray(np.uint8(your_numpy_array))

  # Create a byte stream from the image
  img_byte_arr = BytesIO()
  img.save(img_byte_arr, format='PNG')
  img_byte_arr.seek(0)

  # Use AWS Textract to extract text from the image byte stream
  client = boto3.client('textract', region_name='ap-south-1')
  response = client.detect_document_text(Document={'Bytes': img_byte_arr.read()})

  # Print the extracted text
  # return (response['Blocks'][1]['Text'])
  blocks = response['Blocks']
  if len(blocks) > 1:
    return blocks[1]['Text']
  else:
    return ""
