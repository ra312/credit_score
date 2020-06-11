from azureml.core import Workspace, Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core import Image


ws = Workspace.from_config()
# env = Environment(name = 'sahulat_env')
env = Environment.from_conda_specification(name='az-ls-ds-ml-env',file_path='../az-ls-ds-ml-env.yml')

# conda_dep = CondaDependencies()
# # Installs numpy version 1.17.0 conda package
# conda_deps = ['blas=1.0',
#  'ca-certificates=2020.1.1', 'certifi=2020.4.5.1', 'intel-openmp=2019.4', 'joblib=0.14.1', 'libcxx=4.0.1=h579ed51_0', 
#  'libcxxabi=4.0.1', 'libedit=3.1.20181209', 'libffi=3.2.1', 'libgfortran=3.0.1', 'libsodium=1.0.16', 
#  'llvm-openmp=4.0.1', 'mkl=2019.4', 'mkl-service=2.3.0', 'mkl_fft=1.0.15', 'mkl_random=1.1.0', 'ncurses=6.2', 
#  'numpy=1.18.1', 'numpy-base=1.18.1', 'openssl=1.1.1f', 'pandas=1.0.3', 'pandoc=2.2.3.2', 'pip=20.0.2', 'python=3.8.2',
#   'python-dateutil=2.8.1', 'pytz=2019.3', 'readline=8.0', 'scikit-learn=0.22.1', 'scipy=1.4.1', 'setuptools=46.1.3', 
#   'six=1.14.0', 'sqlite=3.31.1', 'tk=8.6.8', 'wheel=0.34.2', 'xz=5.2.5', 'zlib=1.2.11']
# pip_deps=['adal==1.2.2', 'azure-common==1.1.25','azure-core==1.4.0','azure-graphrbac==0.61.1','azure-identity==1.2.0',
# 'azure-mgmt-authorization==0.60.0','azure-mgmt-containerregistry==2.8.0','azure-mgmt-keyvault==2.2.0','azure-mgmt-resource==8.0.1',
# 'azure-mgmt-storage==9.0.0','azureml==0.2.7','azureml-core==1.3.0.post1','azureml-dataprep==1.4.3','azureml-dataprep-native==14.1.0',
# 'backports-tempfile==1.0','backports-weakref==1.0.','post1cffi==1.14.0','chardet==3.0.4','cloudpickle==1.3.0',
# 'contextlib2==0.6.0.post1','cryptography==2.9d','istro==1.5.0','docker==4.2.0',
# 'dotnetcore2==2.1.13','idna==2.9','importlib-metadata==1.6.0','isodate==0.6.0','jeepney==0.4.3','jmespath==0.9.5',
# 'jsonpickle==1.4',
# 'msal==1.2.0','msal-extensions==0.1.3','msrest==0.6.13','msrestazure==0.6.3','ndg-httpsclient==0.5.1',
# 'oauthlib==3.1.0','pathspec==0.8.0','portalocker==1.7.0','pyasn1==0.4.8','pycparser==2.20','pyjwt==1.7.1','pyopenssl==19.1.0',
# 'requests==2.23.0','requests-oauthlib==1.3.0','ruamel-yaml==0.15.89','secretstorage==3.1.2','urllib3==1.25.9','websocket-client==0.57.0','zipp==3.1.0']

# for package in conda_deps:
#     conda_dep.add_conda_package(package)
# for package in pip_deps:
#     conda_dep.add_pip_package(package)

# env.python.conda_dependencies=conda_dep
env.register(workspace=ws)
build = env.build(workspace=ws)
build.wait_for_completion(show_output=True)
# myenv.register(workspace=ws)


# restored_environment = Environment.get(workspace=ws,name="myenv",version="2")
