{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting streamlitNote: you may need to restart the kernel to use updated packages.\n",
      "  Downloading streamlit-0.67.1-py2.py3-none-any.whl (7.2 MB)\n",
      "Collecting boto3\n",
      "  Downloading boto3-1.15.7-py2.py3-none-any.whl (129 kB)\n",
      "Requirement already satisfied: python-dateutil in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from streamlit) (2.8.1)\n",
      "Requirement already satisfied: protobuf>=3.6.0 in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from streamlit) (3.13.0)\n",
      "Collecting cachetools>=4.0\n",
      "  Using cached cachetools-4.1.1-py3-none-any.whl (10 kB)\n",
      "Collecting botocore>=1.13.44\n",
      "  Downloading botocore-1.18.7-py2.py3-none-any.whl (6.6 MB)\n",
      "Collecting astor\n",
      "  Using cached astor-0.8.1-py2.py3-none-any.whl (27 kB)\n",
      "Collecting watchdog\n",
      "  Downloading watchdog-0.10.3.tar.gz (94 kB)\n",
      "Collecting validators\n",
      "  Downloading validators-0.18.1-py3-none-any.whl (19 kB)\n",
      "Collecting blinker\n",
      "  Downloading blinker-1.4.tar.gz (111 kB)\n",
      "Collecting pyarrow\n",
      "  Downloading pyarrow-1.0.1-cp37-cp37m-win_amd64.whl (10.5 MB)\n",
      "\n",
      "Collecting tzlocal\n",
      "  Using cached tzlocal-2.1-py2.py3-none-any.whl (16 kB)\n",
      "Requirement already satisfied: tornado>=5.0 in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from streamlit) (6.0.4)\n",
      "Collecting base58\n",
      "  Downloading base58-2.0.1-py3-none-any.whl (4.3 kB)\n",
      "Collecting enum-compat\n",
      "  Downloading enum_compat-0.0.3-py3-none-any.whl (1.3 kB)\n",
      "Requirement already satisfied: click>=7.0 in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from streamlit) (7.1.2)\n",
      "Collecting altair>=3.2.0\n",
      "  Downloading altair-4.1.0-py3-none-any.whl (727 kB)\n",
      "Requirement already satisfied: toml in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from streamlit) (0.10.1)\n",
      "Requirement already satisfied: pillow>=6.2.0 in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from streamlit) (7.2.0)\n",
      "Requirement already satisfied: numpy in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from streamlit) (1.19.2)\n",
      "Requirement already satisfied: requests in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from streamlit) (2.24.0)\n",
      "Requirement already satisfied: packaging in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from streamlit) (20.4)\n",
      "Requirement already satisfied: pandas>=0.21.0 in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from streamlit) (1.1.2)\n",
      "Collecting pydeck>=0.1.dev5\n",
      "  Downloading pydeck-0.5.0b1-py2.py3-none-any.whl (4.4 MB)\n",
      "Collecting s3transfer<0.4.0,>=0.3.0\n",
      "  Using cached s3transfer-0.3.3-py2.py3-none-any.whl (69 kB)\n",
      "Collecting jmespath<1.0.0,>=0.7.1\n",
      "  Using cached jmespath-0.10.0-py2.py3-none-any.whl (24 kB)\n",
      "Requirement already satisfied: six>=1.5 in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from python-dateutil->streamlit) (1.15.0)\n",
      "Requirement already satisfied: setuptools in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from protobuf>=3.6.0->streamlit) (49.6.0.post20200925)\n",
      "Requirement already satisfied: urllib3<1.26,>=1.20; python_version != \"3.4\" in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from botocore>=1.13.44->streamlit) (1.25.10)\n",
      "Collecting pathtools>=0.1.1\n",
      "  Downloading pathtools-0.1.2.tar.gz (11 kB)\n",
      "Requirement already satisfied: decorator>=3.4.0 in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from validators->streamlit) (4.4.2)\n",
      "Requirement already satisfied: pytz in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from tzlocal->streamlit) (2020.1)\n",
      "Requirement already satisfied: entrypoints in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from altair>=3.2.0->streamlit) (0.3)\n",
      "Collecting toolz\n",
      "  Downloading toolz-0.11.1-py3-none-any.whl (55 kB)\n",
      "Requirement already satisfied: jinja2 in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from altair>=3.2.0->streamlit) (2.11.2)\n",
      "Requirement already satisfied: jsonschema in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from altair>=3.2.0->streamlit) (3.2.0)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from requests->streamlit) (2020.6.20)\n",
      "Requirement already satisfied: chardet<4,>=3.0.2 in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from requests->streamlit) (3.0.4)\n",
      "Requirement already satisfied: idna<3,>=2.5 in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from requests->streamlit) (2.10)\n",
      "Requirement already satisfied: pyparsing>=2.0.2 in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from packaging->streamlit) (2.4.7)\n",
      "Requirement already satisfied: traitlets>=4.3.2 in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from pydeck>=0.1.dev5->streamlit) (5.0.4)\n",
      "Requirement already satisfied: ipykernel>=5.1.2; python_version >= \"3.4\" in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from pydeck>=0.1.dev5->streamlit) (5.3.4)\n",
      "Requirement already satisfied: ipywidgets>=7.0.0 in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from pydeck>=0.1.dev5->streamlit) (7.5.1)\n",
      "Requirement already satisfied: MarkupSafe>=0.23 in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from jinja2->altair>=3.2.0->streamlit) (1.1.1)\n",
      "Requirement already satisfied: importlib-metadata; python_version < \"3.8\" in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from jsonschema->altair>=3.2.0->streamlit) (2.0.0)\n",
      "Requirement already satisfied: attrs>=17.4.0 in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from jsonschema->altair>=3.2.0->streamlit) (20.2.0)\n",
      "Requirement already satisfied: pyrsistent>=0.14.0 in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from jsonschema->altair>=3.2.0->streamlit) (0.17.3)\n",
      "Requirement already satisfied: ipython-genutils in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from traitlets>=4.3.2->pydeck>=0.1.dev5->streamlit) (0.2.0)\n",
      "Requirement already satisfied: ipython>=5.0.0 in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from ipykernel>=5.1.2; python_version >= \"3.4\"->pydeck>=0.1.dev5->streamlit) (7.18.1)\n",
      "Requirement already satisfied: jupyter-client in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from ipykernel>=5.1.2; python_version >= \"3.4\"->pydeck>=0.1.dev5->streamlit) (6.1.7)\n",
      "Requirement already satisfied: widgetsnbextension~=3.5.0 in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (3.5.1)\n",
      "Requirement already satisfied: nbformat>=4.2.0 in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (5.0.7)\n",
      "Requirement already satisfied: zipp>=0.5 in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from importlib-metadata; python_version < \"3.8\"->jsonschema->altair>=3.2.0->streamlit) (3.2.0)\n",
      "Requirement already satisfied: pickleshare in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from ipython>=5.0.0->ipykernel>=5.1.2; python_version >= \"3.4\"->pydeck>=0.1.dev5->streamlit) (0.7.5)\n",
      "Requirement already satisfied: colorama; sys_platform == \"win32\" in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from ipython>=5.0.0->ipykernel>=5.1.2; python_version >= \"3.4\"->pydeck>=0.1.dev5->streamlit) (0.4.3)\n",
      "Requirement already satisfied: backcall in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from ipython>=5.0.0->ipykernel>=5.1.2; python_version >= \"3.4\"->pydeck>=0.1.dev5->streamlit) (0.2.0)\n",
      "Requirement already satisfied: jedi>=0.10 in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from ipython>=5.0.0->ipykernel>=5.1.2; python_version >= \"3.4\"->pydeck>=0.1.dev5->streamlit) (0.17.2)\n",
      "Requirement already satisfied: pygments in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from ipython>=5.0.0->ipykernel>=5.1.2; python_version >= \"3.4\"->pydeck>=0.1.dev5->streamlit) (2.7.1)\n",
      "Requirement already satisfied: prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0 in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from ipython>=5.0.0->ipykernel>=5.1.2; python_version >= \"3.4\"->pydeck>=0.1.dev5->streamlit) (3.0.7)\n",
      "Requirement already satisfied: jupyter-core>=4.6.0 in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from jupyter-client->ipykernel>=5.1.2; python_version >= \"3.4\"->pydeck>=0.1.dev5->streamlit) (4.6.3)\n",
      "Requirement already satisfied: pyzmq>=13 in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from jupyter-client->ipykernel>=5.1.2; python_version >= \"3.4\"->pydeck>=0.1.dev5->streamlit) (19.0.2)\n",
      "Requirement already satisfied: notebook>=4.4.1 in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (6.1.4)\n",
      "Requirement already satisfied: parso<0.8.0,>=0.7.0 in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from jedi>=0.10->ipython>=5.0.0->ipykernel>=5.1.2; python_version >= \"3.4\"->pydeck>=0.1.dev5->streamlit) (0.7.1)\n",
      "Requirement already satisfied: wcwidth in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0->ipython>=5.0.0->ipykernel>=5.1.2; python_version >= \"3.4\"->pydeck>=0.1.dev5->streamlit) (0.2.5)\n",
      "Requirement already satisfied: pywin32>=1.0; sys_platform == \"win32\" in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from jupyter-core>=4.6.0->jupyter-client->ipykernel>=5.1.2; python_version >= \"3.4\"->pydeck>=0.1.dev5->streamlit) (227)\n",
      "Requirement already satisfied: Send2Trash in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (1.5.0)\n",
      "Requirement already satisfied: terminado>=0.8.3 in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (0.9.1)\n",
      "Requirement already satisfied: nbconvert in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (6.0.6)\n",
      "Requirement already satisfied: prometheus-client in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (0.8.0)\n",
      "Requirement already satisfied: argon2-cffi in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (20.1.0)\n",
      "Requirement already satisfied: pywinpty>=0.5; os_name == \"nt\" in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from terminado>=0.8.3->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (0.5.7)\n",
      "Requirement already satisfied: defusedxml in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (0.6.0)\n",
      "Requirement already satisfied: bleach in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (3.2.1)\n",
      "Requirement already satisfied: testpath in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (0.4.4)\n",
      "Requirement already satisfied: jupyterlab-pygments in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (0.1.2)\n",
      "Requirement already satisfied: mistune<2,>=0.8.1 in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (0.8.4)\n",
      "Requirement already satisfied: pandocfilters>=1.4.1 in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (1.4.2)\n",
      "Requirement already satisfied: nbclient<0.6.0,>=0.5.0 in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (0.5.0)\n",
      "Requirement already satisfied: cffi>=1.0.0 in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (1.14.3)\n",
      "Requirement already satisfied: webencodings in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from bleach->nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (0.5.1)\n",
      "Requirement already satisfied: nest-asyncio in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from nbclient<0.6.0,>=0.5.0->nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (1.4.1)\n",
      "Requirement already satisfied: async-generator in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from nbclient<0.6.0,>=0.5.0->nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (1.10)\n",
      "Requirement already satisfied: pycparser in c:\\users\\hp\\anaconda3\\envs\\pycaret\\lib\\site-packages (from cffi>=1.0.0->argon2-cffi->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets>=7.0.0->pydeck>=0.1.dev5->streamlit) (2.20)\n",
      "Building wheels for collected packages: watchdog, blinker, pathtools\n",
      "  Building wheel for watchdog (setup.py): started\n",
      "  Building wheel for watchdog (setup.py): finished with status 'done'\n",
      "  Created wheel for watchdog: filename=watchdog-0.10.3-py3-none-any.whl size=73876 sha256=bef84b8d679b911cfc5fc8306658cb636a2974de40ecd9deb35e0b57bb97f87d\n",
      "  Stored in directory: c:\\users\\hp\\appdata\\local\\pip\\cache\\wheels\\27\\21\\35\\9d1e531f9de5335147dbef07e9cc99d312525ba128a93d1225\n",
      "  Building wheel for blinker (setup.py): started\n",
      "  Building wheel for blinker (setup.py): finished with status 'done'\n",
      "  Created wheel for blinker: filename=blinker-1.4-py3-none-any.whl size=13454 sha256=3195e4cda5a2f12a64703b77b0466be20a68003ab4a6ebdb531d9ad681183696\n",
      "  Stored in directory: c:\\users\\hp\\appdata\\local\\pip\\cache\\wheels\\22\\f5\\18\\df711b66eb25b21325c132757d4314db9ac5e8dabeaf196eab\n",
      "  Building wheel for pathtools (setup.py): started\n",
      "  Building wheel for pathtools (setup.py): finished with status 'done'\n",
      "  Created wheel for pathtools: filename=pathtools-0.1.2-py3-none-any.whl size=8790 sha256=a73a79952cbd5768789673738031559c00b2cb6ce024200e2cb7fe89eba4fb18\n",
      "  Stored in directory: c:\\users\\hp\\appdata\\local\\pip\\cache\\wheels\\3e\\31\\09\\fa59cef12cdcfecc627b3d24273699f390e71828921b2cbba2\n",
      "Successfully built watchdog blinker pathtools\n",
      "Installing collected packages: jmespath, botocore, s3transfer, boto3, cachetools, astor, pathtools, watchdog, validators, blinker, pyarrow, tzlocal, base58, enum-compat, toolz, altair, pydeck, streamlit\n",
      "Successfully installed altair-4.1.0 astor-0.8.1 base58-2.0.1 blinker-1.4 boto3-1.15.7 botocore-1.18.7 cachetools-4.1.1 enum-compat-0.0.3 jmespath-0.10.0 pathtools-0.1.2 pyarrow-1.0.1 pydeck-0.5.0b1 s3transfer-0.3.3 streamlit-0.67.1 toolz-0.11.1 tzlocal-2.1 validators-0.18.1 watchdog-0.10.3\n"
     ]
    }
   ],
   "source": [
    "pip install streamlit"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Transformation Pipeline and Model Successfully Loaded\n"
     ]
    }
   ],
   "source": [
    "from pycaret.classification import load_model, predict_model\n",
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "model = load_model('catboost_final')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "def welcome():\n",
    "    return \"Welcome All\"\n",
    "\n",
    "\n",
    "def Fraud_Detection(Time,Amount):\n",
    "    \n",
    "\n",
    "   \n",
    "    prediction=classifier.predict([[Time,Amount]])\n",
    "    print(prediction)\n",
    "    return prediction\n",
    "\n",
    "\n",
    "\n",
    "def main():\n",
    "    st.title(\"Fraud Detection\")\n",
    "    html_temp = \"\"\"\n",
    "    <div style=\"background-color:tomato;padding:10px\">\n",
    "    <h2 style=\"color:white;text-align:center;\">Fraud Detection ML App </h2>\n",
    "    </div>\n",
    "    \"\"\"\n",
    "    st.markdown(html_temp,unsafe_allow_html=True)\n",
    "    Time = st.text_input(\"Time\",\"Type Here\")\n",
    "    Amount = st.text_input(\"Amount\",\"Type Here\")\n",
    "    result=\"\"\n",
    "    if st.button(\"Predict\"):\n",
    "        result=Fraud_Detection(Time,Amount)\n",
    "    st.success('The output is {}'.format(result))\n",
    "    if st.button(\"About\"):\n",
    "        st.text(\"Lets LEarn\")\n",
    "        st.text(\"Built with Streamlit\")\n",
    "\n",
    "if __name__=='__main__':\n",
    "    main()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}