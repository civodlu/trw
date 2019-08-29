import glob
import os
import webbrowser

import nox

torch_version = [
    'torch==1.2.0+cpu', 'torchvision==0.4.0+cpu', '-f', 'https://download.pytorch.org/whl/torch_stable.html'
]

def find_files(root_folder, extension):
    """
    Find all files with a specific extension
    """
    files = [f for f in glob.glob(root_folder + '/**/*' + extension, recursive=True)]
    return files
 
@nox.session(reuse_venv=True)
def tests(session):
    # run all tests
    session.install(*torch_version)
    
    session.install('pytest')
    session.install('pytest-cov')
    session.install('.')

    session.run('pytest', '--cov-report', 'term', '--cov-report', 'html', '--cov', 'trw')

@nox.session
def beautify(session):
    # sort the includes
    session.install('isort')
    session.run('isort', '-rc', '.')

    # reformat to code guidelines
    session.install('black')
    session.run('black', '--verbose', 'src')

@nox.session
def lint(session):
    session.install('flake8')
    session.run('flake8', 'src')

@nox.session(reuse_venv=True)
def docs(session):
    #session.install("-r", "requirements-dev.txt")
    #session.install(".")
    #session.install(*torch_version)

    # build the documentation
    session.run('sphinx-build', 'docs/source', 'docs/build')

    # open the documentation in a browser
    path = 'file://' + os.path.realpath('docs/build/index.html')
    webbrowser.open_new(path)

    # run the python commands contained in the docstrings
    python_files = find_files('src', extension='.py')
    print('Python files:\n', python_files)
    for file in python_files:
        if not '__init__.py' in file:
            session.run('python', '-m', 'doctest', '-v', file)

@nox.session
def publish(session):
   session.install('twine')
   session.run('python', 'setup.py', 'bdist_wheel')
   # TODO remove exsiting build/dist
   session.run('twine', 'upload', 'dist/*')
