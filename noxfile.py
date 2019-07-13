import nox
import webbrowser
import os
import glob

def find_files(root_folder, extension):
    """
    Find all files with a specific extension
    """
    files = [f for f in glob.glob(root_folder + '/**/*' + extension, recursive=True)]
    return files

@nox.session
def tests(session):
    # run all tests
    session.install('pytest')
    session.install('.')
    session.run('pytest')

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
