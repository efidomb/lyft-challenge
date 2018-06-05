import shutil
import os
from upload import TransferData

files_folder = 'files'
if not os.path.exists(files_folder):
    os.makedirs(files_folder)
files = ['main.py', 'helper.py', 'project_tests.py', 'mydemo.py', 'upload.py', 'upload_files.py']
for file in files:
	shutil.copyfile(file, os.path.join(files_folder, file))
transferData = TransferData()
transferData.upload_file(files_folder, delete_original=True)