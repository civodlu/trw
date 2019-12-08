"""
Careful here: we MUST use ordered set as order of execution is important! If not there may be inconsistencies
between compiled net and executed net
"""
from enum import Enum
import collections
import torch.nn as nn
import weakref
from trw.simple_layers import SimpleLayerBase, SimpleOutputBase
from trw.simple_layers.simple_layers_implementations import Input
from trw.simple_layers.ordered_set import OrderedSet


def find_layer_type(nodes: list, layer_type):
    """
    Find all layer with a given type

    Args:
        nodes: the starting nodes [list]
        layer_type: the type of the nodes to collect
    Returns:
        a list of nodes of the corresponding type
    """
    assert isinstance(nodes, list), 'must be a list!'
    for node in nodes:
        assert isinstance(node, SimpleLayerBase), 'must be a layer'

    visited_nodes = OrderedSet()
    next_nodes = nodes.copy()  # prevent side effects!
    collected_nodes = OrderedSet()

    while len(next_nodes):
        current_node = next_nodes.pop()
        visited_nodes.add(current_node)

        if isinstance(current_node, layer_type):
            collected_nodes.add(current_node)

        for child_ref in current_node.children:
            child = child_ref()
            assert child is not None, 'some part of the network was deleted, make sure the output nodes are saved!'
            if child not in visited_nodes:
                next_nodes.append(child)

        for parent in current_node.parents:
            if parent not in visited_nodes:
                next_nodes.append(parent)

    return list(collected_nodes)


def nodes_mark_output_dependencies(output_nodes: list):
    """
    Marks nodes by output IDs

    Args:
        output_nodes: a list of output nodes to be marked

    Returns:
        nodes with a set of output IDs
    """
    assert isinstance(output_nodes, list)
    for node in output_nodes:
        assert isinstance(node, SimpleLayerBase)

    marks_by_node = collections.defaultdict(set)
    output_mapping = {}
    for index, output_node in enumerate(output_nodes):
        output_mapping[output_node] = index
        marks_by_node[output_node].add(index)

    visited_nodes = OrderedSet()
    next_nodes = output_nodes.copy()  # prevent side effects!
    while len(next_nodes):
        current_node = next_nodes.pop()
        current_marks = marks_by_node[current_node]
        visited_nodes.add(current_node)

        for parent in current_node.parents:
            marks_by_node[parent].update(current_marks)
            if parent not in visited_nodes:
                next_nodes.append(parent)

    return dict(marks_by_node)


def remove_weak_ref(nodes: list):
    """
    Run through all the nodes of the graph and remove `weakref`.

    `weakref` are an issue when importing or exporting the network with `pickle`.
        We can safely remove these `weakref` and reconstruct them if necessary.

    Args:
        nodes: the starting nodes [list]

    Returns:
        None
    """
    assert isinstance(nodes, list), 'must be a list!'
    for node in nodes:
        assert isinstance(node, SimpleLayerBase), 'must be a layer'

    visited_nodes = OrderedSet()
    next_nodes = nodes.copy()  # prevent side effects!

    while len(next_nodes):
        current_node = next_nodes.pop()
        visited_nodes.add(current_node)

        for child_ref in current_node.children:
            child = child_ref()
            assert child is not None, 'some part of the network was deleted, make sure the output nodes are saved!'
            if child not in visited_nodes:
                next_nodes.append(child)

        for parent in current_node.parents:
            if parent not in visited_nodes:
                next_nodes.append(parent)

        current_node.children = OrderedSet()


def create_weak_ref(output_nodes: list):
    """
    Re-create the `weakref` references of the children for the given `output_nodes` recursively

    Args:
        output_nodes: a list of output nodes to have the children weak ref updated

    Returns:
        None
    """
    assert isinstance(output_nodes, list)
    for node in output_nodes:
        assert isinstance(node, SimpleLayerBase)

    visited_nodes = OrderedSet()
    next_nodes = output_nodes.copy()  # prevent side effects!
    while len(next_nodes):
        current_node = next_nodes.pop()
        visited_nodes.add(current_node)

        for parent in current_node.parents:
            parent.children.add(weakref.ref(current_node))

            if parent not in visited_nodes:
                next_nodes.append(parent)


class RuntimeAction(Enum):
    """
    Specifies the type of runtime action (e.g., execution of a node, release of node's state or evaluation of a node)
    """
    EXECUTE_NODE = 1
    REMOVE_STATE = 2
    EVALUATE_STATE = 3


class CompiledNet(nn.Module):
    """
    Encapsulate a `compiled` network so that we can efficiently calculate the outputs
        of the network.
    """
    def __init__(self):
        super().__init__()

        self.inputs = None
        self.outputs = None
        self.runtime_actions = []
        self.torch_modules = None

        # here we may only want to evaluate a partial network but still keep alive the
        # rest of the network. So here we can store the outputs that are not
        # directly calculated by the compiled net
        self.other_outputs_to_keep_alive = None

        # we NEED to keep this module list for torch:
        # it means torch is aware of the parameters of our sub-modules
        self.module_list = nn.ModuleList()
        
    def collect_parameters(self):
        """
        Make the parameters of each node visible to the current module
        """
        self.torch_modules = nn.ModuleList()
        added_modules_or_nodes = OrderedSet()
        for action_type, node in self.runtime_actions:
            if node not in added_modules_or_nodes:
                module = node.get_module()
                if module is not None and module not in added_modules_or_nodes:
                    if isinstance(module, nn.Module):
                        # if the module is not a `nn.Module`, then it should be a functional pytorch module
                        self.torch_modules.append(module)
                    added_modules_or_nodes.add(module)

                added_modules_or_nodes.add(node)

    def forward(self, batch):
        """
        Calculate the outputs of a network

        Args:
            batch: (dict) a dictionary like of features

        Returns:
            a dictionary of outputs
        """
        states = {}
        outputs = collections.OrderedDict()

        def prepare_inputs(action):
            action_inputs = []
            for parent in action.parents:
                assert parent in states, 'TODO DEBUG, input_not_found=' + str(parent)
                input_value = states.get(parent)
                assert input_value is not None, 'TODO bug! The parents were not calculated or their state was released too early!'
                action_inputs.append(input_value)
            if len(action_inputs) == 1:
                action_inputs = action_inputs[0]  # TODO evaluate this! should we keep the list of size 1 or not?
            return action_inputs

        # feed the requested inputs
        for i in self.inputs:
            #print('FETCH Input=', i)
            input_value = batch.get(i.feature_name)
            assert input_value is not None, 'feature `{}` is missing from the batch. Requested by node={}'.format(i.feature_name, i)
            states[i] = input_value

        # perform the actions
        for action_type, action in self.runtime_actions:
            if action_type == RuntimeAction.EXECUTE_NODE:
                #print('EXECUTE=', action, 'state_size=', len(states), 'parents=', len(action.parents))
                action_inputs = prepare_inputs(action)
                module = action.get_module()
                assert module is not None, 'module is not created!'

                action_output = module(action_inputs)

                # just for safety, make sure our estimated output size
                # is the same as the one calculated
                predicted_shape = action.shape[1:]
                actual_shape = list(action_output.shape)[1:]
                assert predicted_shape == actual_shape, f'shape mismatch between simple_layer predicted ' \
                                                        f'shape ({predicted_shape}) and actual shape ' \
                                                        f'({actual_shape}) for module={module}'
                states[action] = action_output
            elif action_type == RuntimeAction.REMOVE_STATE:
                #print('del STATE=', action)
                del states[action]
            elif action_type == RuntimeAction.EVALUATE_STATE:
                #print('EVAL=', action)
                action_inputs = prepare_inputs(action)
                evaluation = action.forward(action_inputs, batch)
                outputs[action.output_name] = evaluation
            else:
                assert 0, 'action_type={} is not handled!'.format(action_type)

        return outputs

    def __getstate__(self):
        # `weakref` can't be serialized: we need to manually remove them
        # and reconstruct them later
        remove_weak_ref(self.outputs + self.inputs)
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state
        # before serialization, the `weakref` were destroyed. Now we need to re-create them!
        create_weak_ref(self.outputs)


def compile_nn(output_nodes: list, other_outputs_to_keep_alive=None):
    """
    Compile a network to calculate `output_nodes`

    TODO:

        * graph optimizations:

            * transform linear nodes as a single `sequence`!

    Args:
        output_nodes: the output nodes to calculate. The order of the nodes indicates the order of the calculation
            and impacts the book-keeping of the shared calculation in multiple output networks
        other_outputs_to_keep_alive: keeps alive unused output nodes

    Returns:
        a `CompiledNet`
    """
    assert isinstance(output_nodes, list), 'must be a list'

    # make sure we don't have outputs with the same name
    outputs_names = set()
    for output in output_nodes:
        assert isinstance(output, SimpleOutputBase), 'output must be a trw.simple_layers.SimpleOutputBase'
        name = output.output_name
        assert name not in outputs_names, 'output with the same name:{}, node={}'.format(name, str(output))
        outputs_names.add(name)

    input_nodes = find_layer_type(output_nodes, Input)

    # here we record all nodes reachable from the output
    # indeed, we may not be interested in all inputs
    # if some calculations are dis-joint. All visited nodes MUST
    # have an output mark
    dependencies = nodes_mark_output_dependencies(output_nodes)

    # this list contains the nodes that are currently kept in memory
    # to be used by the network at a later state (e.g., if we have a
    # multiple outputs for one node)
    node_bookkeeping = collections.defaultdict(set)

    # now we need to figure out the order of the calculations: proceed by depth-first
    # so that we limit the number of node book beeping
    visited_nodes = set()
    next_nodes = input_nodes.copy()  # prevent side effects!

    # a list of actions to be performed to calculate the specified `output_nodes`
    actions = []

    def release_node(node_to_release):
        # free book-keeping
        actions.append((RuntimeAction.REMOVE_STATE, node_to_release))

    def visit_nodes(current_node):
        if current_node not in visited_nodes:
            visited_nodes.add(current_node)
            if not isinstance(current_node, Input):
                if isinstance(current_node, SimpleOutputBase):
                    actions.append((RuntimeAction.EVALUATE_STATE, current_node))
                else:
                    actions.append((RuntimeAction.EXECUTE_NODE, current_node))

            for parent in current_node.parents:
                parent_book_keeping = node_bookkeeping.get(parent)
                if parent_book_keeping is not None:
                    # we visited a child node so we can remove the parent's dependency
                    parent_book_keeping.discard(current_node)

                    if len(parent_book_keeping) == 0:
                        # all dependencies are gone! free the book_keeping for this node
                        release_node(parent)

            # node dependencies: unless all children nodes were deleted,
            # we can't clear a temporary calculation
            s = OrderedSet()
            for child_ref in current_node.children:
                child = child_ref()
                assert child is not None, 'child node was deleted!'
                s.add(child)
            node_bookkeeping[current_node] = s

    while len(next_nodes):
        current_node = next_nodes.pop()

        # first, make sure we have all the dependencies to calculate the node (e.g., multiple parent for one node)
        # else, move the node down the list
        can_be_visited = True
        if current_node not in visited_nodes:
            for parent in current_node.parents:
                book_keeping = node_bookkeeping.get(parent)
                if book_keeping is None:
                    next_nodes.insert(0, current_node)
                    can_be_visited = False
                    break
                    
        if not can_be_visited:
            # the node can't be visisted yet (e.g., one parent doesn't its state calculated
            # so let's try next ones!
            continue

        visit_nodes(current_node)

        for child_ref in current_node.children:
            child = child_ref()
            assert child is not None, 'some part of the network was deleted, make sure the output nodes are saved!'
            if child not in visited_nodes:
                # visit only the nodes required to be calculated for the given
                # `output_nodes`
                marks = dependencies.get(child)
                if marks is not None and len(marks) > 0:
                    next_nodes.append(child)

    compiled_net = CompiledNet()
    compiled_net.inputs = input_nodes
    compiled_net.outputs = output_nodes
    compiled_net.runtime_actions = actions
    compiled_net.other_outputs_to_keep_alive = other_outputs_to_keep_alive
    compiled_net.collect_parameters()

    return compiled_net
