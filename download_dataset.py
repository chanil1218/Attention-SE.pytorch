#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
if 'google_drive_downloader' not in sys.modules:
    get_ipython().system('pip install googledrivedownloader')
from google_drive_downloader import GoogleDriveDownloader as gdd


# In[2]:


dest_path = 'home/sangwon/바탕화면/dataset'
train_path = dest_path + "/train.tar.gz"
valid_path = dest_path + "/valid.tar.gz"
test_path = dest_path + "/test.tar.gz"


# In[ ]:


# Download Train set
gdd.download_file_from_google_drive(file_id='1CULVCAq0T3wqZTPGIqPja6OtwjYJkGAy',
                                    dest_path='dataset/train.tar.gz',
                                    unzip=False, showsize=True)
get_ipython().system('tar xvzf dataset/train.tar.gz')


# In[ ]:


# Download Valid set
gdd.download_file_from_google_drive(file_id='1WE229Jt9WV2iZbxY7YjkYIfSZyCHz9Iq',
                                    dest_path='dataset/valid.tar.gz',
                                    unzip=False, showsize=True)
get_ipython().system('tar xvzf dataset/valid.tar.gz')


# In[ ]:


# Download Test set
gdd.download_file_from_google_drive(file_id='1mERBbcwBgGjRpHeGf3d871DEW4EvI7fi',
                                    dest_path='dataset/valid.tar.gz',
                                    unzip=False, showsize=True)
get_ipython().system('tar xvzf dataset/test.tar.gz')


# In[ ]:


ls


# In[ ]:




