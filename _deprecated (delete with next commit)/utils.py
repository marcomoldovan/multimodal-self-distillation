import os

from datasets import load_dataset, load_from_disk

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

count_parameters = lambda model : {'requires_grad':sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6,
                                   'does_not_require_grad':sum(p.numel() for p in model.parameters() if not p.requires_grad)/1e6}



def freeze_whole_model(model):
  for param in model.parameters():
    param.requires_grad = False
    
    
def freeze_model_except_last_n_layers(speech_model=None, text_model=None, train_last_n_speech_model_layers=0, train_last_n_text_model_layers=0):
  # freeze all layers in speech model
  if speech_model is not None and train_last_n_speech_model_layers == 0:
    freeze_whole_model(speech_model)
    
  # freeze all except last n in speech model 
  elif speech_model is not None and train_last_n_speech_model_layers > 0:
    if hasattr(speech_model, 'feature_extractor'):
      for param in speech_model.feature_extractor.parameters():
        param.requires_grad = False
    for param in speech_model.feature_projection.parameters():
      param.requires_grad = False
    for param in speech_model.encoder.pos_conv_embed.parameters():
      param.requires_grad = False
    for param in speech_model.encoder.layer_norm.parameters():
      param.requires_grad = False
    for param in speech_model.encoder.dropout.parameters():
      param.requires_grad = False
    for i, encoder_layer in enumerate(speech_model.encoder.layers._modules):
      if i < (len(speech_model.encoder.layers._modules) - train_last_n_speech_model_layers):
        for param in speech_model.encoder.layers[i].parameters():
          param.requires_grad = False
  
  # freeze all layers in text model  
  if text_model is not None and train_last_n_text_model_layers == 0:
    freeze_whole_model(text_model)
  
  # freeze all except last n in text model
  elif text_model is not None and train_last_n_text_model_layers > 0:
    for param in text_model.embeddings.parameters():
      param.requires_grad = False
    for i, encoder_layer in enumerate(text_model.encoder.layer._modules):
      if i < (len(text_model.encoder.layer) - train_last_n_text_model_layers):
        for param in text_model.encoder.layer[i].parameters():
          param.requires_grad = False
    
    
    
def assert_type_of_machine():
  try:
    import google.colab
    on_colab = True
  except:
    on_colab = False
    
  if os.popen('hostname').read()[:-1] == 'DESKTOP-3EBED7S':
    on_pc = True
  else:
    on_pc = False
    
  if not on_colab and not on_pc:
    on_server = True
  else:
    on_server = False
    
    
  return on_colab, on_pc, on_server


def assert_presence_of_dataset_on_machine(dataset='librispeech'):
  """
  Returns True if the dataset is present on the machine or accesible from 
  within Colab. Also returns the relative path to the dataset for further
  use in load_from_disk() given that the dataset is on the machine.
  """
  on_colab, on_pc, on_server = assert_type_of_machine()

  preprocessed_data_available_on_machine = False
  
  if on_colab:
    data_path = f'/content/drive/MyDrive/Projects/Cross-Modal Speech Segment Retrieval/cross-modal-speech-segment-retrieval/data/{dataset}/'
    if os.path.isdir(data_path):
      preprocessed_data_available_on_machine = True
  elif on_pc or on_server:
    if os.path.isfile(f'./data/{dataset}/train/0/dataset.arrow'):
      preprocessed_data_available_on_machine = True
      data_path = f'./data/{dataset}/'
    elif os.path.isfile(f'E:/Machine Learning/Datasets/{dataset}/train/0/dataset.arrow'):
      preprocessed_data_available_on_machine = True
      data_path = f'E:/Machine Learning/Datasets/{dataset}/'

  return preprocessed_data_available_on_machine, data_path


def assert_presence_of_dataset_on_gdrive(dataset='librispeech'):
  #TODO modify as to check whether a preprocessed version of the dataset is available on Google Drive
  #TODO with throw FileNotFoundError: [Errno 2] No such file or directory: 'credentials.json' <-- deal with credentials.json
  #source: https://developers.google.com/drive/api/quickstart/python
  #help for credentials: https://developers.google.com/workspace/guides/create-credentials#desktop-app
  #help for checking whether file exsits: https://stackoverflow.com/questions/56496333/google-drive-api-check-if-folder-exists
  """Shows basic usage of the Drive v3 API.
  Prints the names and ids of the first 10 files the user has access to.
  """
  return False, ''
  
  # If modifying these scopes, delete the file token.json.
  SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly']
  
  creds = None
  # The file token.json stores the user's access and refresh tokens, and is
  # created automatically when the authorization flow completes for the first
  # time.
  if os.path.exists('token.json'):
      creds = Credentials.from_authorized_user_file('token.json', SCOPES)
  # If there are no (valid) credentials available, let the user log in.
  if not creds or not creds.valid:
      if creds and creds.expired and creds.refresh_token:
          creds.refresh(Request())
      else:
          flow = InstalledAppFlow.from_client_secrets_file(
              'credentials.json', SCOPES)
          creds = flow.run_local_server(port=0)
      # Save the credentials for the next run
      with open('token.json', 'w') as token:
          token.write(creds.to_json())

  try:
      service = build('drive', 'v3', credentials=creds)

      # Call the Drive v3 API
      results = service.files().list(
          pageSize=10, fields="nextPageToken, files(id, name)").execute()
      items = results.get('files', [])

      if not items:
          print('No files found.')
          return
      print('Files:')
      for item in items:
          print(u'{0} ({1})'.format(item['name'], item['id']))
  except HttpError as error:
      # TODO(developer) - Handle errors from drive API.
      print(f'An error occurred: {error}')
    

def download_dataset_from_gdrive(dataset='librispeech'):
    """Downloads the dataset to the machine from Google Drive."""
    #https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    #https://stackoverflow.com/questions/52619878/python-how-do-download-entire-folder-from-google-drive
    #https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    pass
    
    
def assert_presence_of_dataset_on_onedrive(dataset='librispeech'):
    pass
  
  
def download_dataset_from_onedrive(dataset='librispeech'):
    pass
  