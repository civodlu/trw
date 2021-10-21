Conda requirement file for readthedocs
--------------------------------------

we use conda cpu to limit the size of the download else we reach the 1gb limit on RTD.

Mypy
----

To run mypy:

::

    mypy src --allow-redefinition