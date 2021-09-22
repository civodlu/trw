from torch import nn


def _get_decorators(function):
    """
    Return the decorators of a function

    This was adapted from:
        https://schinckel.net/2012/01/20/get-decorators-wrapping-a-function/

    However, this does not work for many type of decorators... so should only
    be used with <is_autocast_module_decorated>
    """
    # If we have no __closure__, it means we are not wrapping any other functions.
    if not hasattr(function, '__closure__') or not function.__closure__:
        return [function]

    # Otherwise, we want to collect all of the recursive results for every closure we have.
    decorators = []
    for cell in function.__closure__:
        if hasattr(cell.cell_contents, '__call__'):
            decorators.extend(_get_decorators(cell.cell_contents))
    return [function] + decorators


def is_autocast_module_decorated(module: nn.Module):
    """
    Return `True` if a nn.Module.forward was decorated with
    torch.cuda.amp.autocast
    """
    try:
        from torch.cuda.amp import autocast
        decorators = _get_decorators(module.forward)
        for d in decorators:
            if isinstance(d, autocast):
                return True
    except:
        pass

    return False
