import math

RED, BLACK, DOUBLEBLACK = 0, 1, 2

class Node:
    def __init__(self, val, parent=None, left=None, right=None) -> None:
        self.parent = parent
        self.val = val
        self.left = left
        self.right = right

class WeightedNode(Node):
    def __init__(self, val, weight=0, parent=None, left=None, right=None) -> None:
        super().__init__(val, parent, left, right)
        self.weight = weight
        self.sum = weight

    def update_sum(self):
        self.sum = self.weight + self.left.sum + self.right.sum

class ColorWeightedNode(WeightedNode):
    def __init__(self, val, color=BLACK, weight=0, parent=None, left=None, right=None) -> None:
        super().__init__(val, weight, parent, left, right)
        self.color = color



class BST:
    def __init__(self, node_cls=Node, debug=False) -> None:
        self.nil = node_cls(None)
        self.root = self.nil
        self.debug = debug

    def _check_properties(self):
        if not self.debug: return True
        def recurse(node: Node):
            if node == self.nil: return True
            assert node.val is not None
            if node.left != self.nil:
                assert node.left.val < node.val, f"left={node.left.val} > parent={node.val}"
            if node.right != self.nil:
                assert node.right.val > node.val, f"right={node.right.val} < parent={node.val}"
            recurse(node.left)
            recurse(node.right)

        recurse(self.root)

        assert self.nil.val is None
        assert self.nil.left is None
        assert self.nil.right is None
        # print("BST Checks")
        return True

    def __search_tree_helper(self, node, key):
        if node == self.nil or key == node.val:
            return node

        if key < node.val:
            return self.__search_tree_helper(node.left, key)
        return self.__search_tree_helper(node.right, key)

    # search the tree for the key k
    # and return the corresponding node
    def searchTree(self, k):
        return self.__search_tree_helper(self.root, k)

    # find the node with the minimum key
    def minimum(self, node):
        while node.left != self.nil:
            node = node.left
        return node

    # find the node with the maximum key
    def maximum(self, node):
        while node.right != self.nil:
            node = node.right
        return node

    # find the successor of a given node
    def successor(self, x):
        # if the right subtree is not None,
        # the successor is the leftmost node in the
        # right subtree
        if x.right != self.nil:
            return self.minimum(x.right)

        # else it is the lowest ancestor of x whose
        # left child is also an ancestor of x.
        y = x.parent
        while y is not None and y != self.nil and x == y.right:
            x = y
            y = y.parent
        if y is None: y = self.nil
        return y

    # find the predecessor of a given node
    def predecessor(self,  x):
        # if the left subtree is not None,
        # the predecessor is the rightmost node in the
        # left subtree
        if (x.left != self.nil):
            return self.maximum(x.left)

        y = x.parent
        while y is not None and y != self.nil and x == y.left:
            x = y
            y = y.parent
        if y is None: y = self.nil
        return y

class WeightedBST(BST):
    def __init__(self, node_cls=WeightedNode, debug=False) -> None:
        super().__init__(node_cls, debug)
        self._eps = 1e-8

    def _check_properties(self):
        if not self.debug: return True
        # weight
        super()._check_properties()
        def check_weight(node: WeightedNode):
            if node == self.nil: return True
            if node.left == self.nil and node.right == self.nil:
                return True

            # assert node.sum == node.left.sum + node.right.sum + node.weight, f"Weight does not sum up: {node.sum} != {node.left.sum} + {node.weight} + {node.right.sum}"
            assert node.sum == node.left.sum + node.right.sum + node.weight, f"Weight does not sum up: {node.sum:.3f} != {node.left.sum:.3f} + {node.weight:.3f} + {node.right.sum:.3f} = {node.left.sum + node.weight + node.right.sum:.3f}"
            check_weight(node.left)
            check_weight(node.right)
        check_weight(self.root)
        assert self.nil.weight == 0 == self.nil.sum, "Nil should have no weight"
        # print("WBST Checks")

    def _update_parent_sum(self, node: WeightedNode):
        while node is not None and node != self.nil:
            node.update_sum()
            node = node.parent

    def query_sum(self, val, inclusive=False):
        def recurse(node):
            if node == self.nil: return 0
            if node.val < val: return node.weight + node.left.sum + recurse(node.right)
            if node.val > val: return recurse(node.left)
            return (node.weight if inclusive else 0) + recurse(node.left)
        return recurse(self.root)

    def query_cumu_weight(self, w, prev=True):
        def recurse(node, w):
            w = max(w, 0.) # adjust for numerical issue
            assert node.sum > w - self._eps
            if node.left.sum <= w and w < node.left.sum + node.weight:
                return node
            if node.right == self.nil and (node.left.sum <= w and w < node.left.sum + node.weight + self._eps):
                return node
            if w < node.left.sum:
                return recurse(node.left, w)
            else: # w >= node.left.sum + node.weight:
                assert node.right != self.nil
                return recurse(node.right, w - node.left.sum - node.weight)
        if w >= self.root.sum:
            node = self.maximum(self.root)
        else:
            node = recurse(self.root, w)
            if prev: node = self.predecessor(node)
        return -math.inf if node == self.nil else node.val

        """
        Assume the threshold to cost is:
            [(0.1, 1), (0.2, 2), (0.4, 3), (0.5, 7)]
        The mass for each threshold is
            [(0.1, 1), (0.2, 1), (0.4, 1), (0.5, 4)]
        Suppose we want to look up is for cost of 3 (i.e. 0.4)
        Suppose we want to look up cost 4 (we would need 0.4 still)
        This means the range for each threshold is
            -inf: [0, 1)
            0.1 : [1, 2)
            0.2 : [2, 3)
            0.4 : [3, 7)
            0.5 : [7, infty)
        We thus need to find the immediate predecesor


        """
# We first assume all entries are different.
# This could be handled by the doubly linked list
class QuantileTree(WeightedBST):
    def __init__(self, node_cls=ColorWeightedNode, debug=False):
        super().__init__(node_cls, debug)

    def _check_properties(self):
        if not self.debug: return True
        super()._check_properties()
        def _check_rb(node: ColorWeightedNode):
            if node == self.nil: return True
            if node.color != RED and node.color != BLACK: return False
            return _check_rb(node.left) and _check_rb(node.right)
        assert _check_rb(self.root), "Not all nodes are red and black."
        assert self.root.color == BLACK, "Root color is wrong"
        assert self.nil.color == BLACK, "Nil color is wrong"
        def _check_rr(node: ColorWeightedNode):
            if node == self.nil: return True
            if node.color == RED:
                if node.left.color == RED or node.right.color == RED: return False
            return _check_rr(node.left) and _check_rr(node.right)
        assert _check_rr(self.root), "Some red nodes have red children"
        def _check_bd(root: ColorWeightedNode):
            def _recurse(curr: ColorWeightedNode, num_black: int):
                if curr == self.nil: return num_black, num_black
                if curr.color == BLACK: num_black += 1
                l_min, l_max = _recurse(curr.left, num_black)
                r_min, r_max = _recurse(curr.right, num_black)
                return min(l_min, r_min), max(l_max, r_max)
            min_cnt, max_cnt = _recurse(root, 0)
            return min_cnt == max_cnt
        assert _check_bd(self.root), "Paths with diff # of blacks"

    def __rb_transplant(self, u, v):
        if u.parent == None:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        par = v.parent = u.parent
        return par

    def __fix_delete(self, x):
        while x != self.root and x.color == BLACK:
            if x == x.parent.left:
                s = x.parent.right
                if s.color == RED:
                    # case 3.1
                    s.color = BLACK
                    x.parent.color = RED
                    self.rotate_left(x.parent)
                    s = x.parent.right
                if s.left.color == s.right.color == BLACK:
                    # case 3.2
                    s.color = RED
                    x = x.parent
                else:
                    if s.right.color == BLACK: #left is red
                        # case 3.3
                        s.left.color = BLACK
                        s.color = RED
                        self.rotate_right(s)
                        s = x.parent.right
                    # case 3.4
                    s.color = x.parent.color
                    x.parent.color = BLACK
                    s.right.color = BLACK
                    self.rotate_left(x.parent)
                    x = self.root
            else:
                s = x.parent.left
                if s.color == RED:
                    # case 3.1
                    s.color = BLACK
                    x.parent.color = RED
                    self.rotate_right(x.parent)
                    s = x.parent.left
                if s.left.color == s.right.color == BLACK:
                    # case 3.2
                    s.color = RED
                    x = x.parent
                else:
                    if s.left.color == BLACK: #right is red
                        # case 3.3
                        s.right.color = BLACK
                        s.color = RED
                        self.rotate_left(s)
                        s = x.parent.left
                    # case 3.4
                    s.color = x.parent.color
                    x.parent.color = BLACK
                    s.left.color = BLACK
                    self.rotate_right(x.parent)
                    x = self.root
        x.color = BLACK

    def delete(self, val, weight=1):
        # find the node containing key
        node = self.root
        z = self.nil
        to_del = []
        while node != self.nil:
            to_del.append(node)
            if node.val < val:
                node = node.right
            elif node.val > val:
                node = node.left
            else:
                z = node
                break
            #if node.val == val:
            #    z = node
            ##TODO: change to <?
            #node = node.right if node.val <= val else node.left

        if z == self.nil:
            raise ValueError("Couldn't find key in the tree")
            return
        if z.weight < weight:
            raise ValueError("Too much weight to subtract")
        for _ in to_del: _.sum -= weight
        if z.weight > weight:
            z.weight -= weight
            return
        assert z.weight == weight # remove the whole node

        y = z
        y_original_color = y.color
        if z.left == self.nil:
            x = z.right
            self.__rb_transplant(z, z.right)
        elif z.right == self.nil:
            x = z.left
            self.__rb_transplant(z, z.left)
        else:
            y = self.minimum(z.right)
            y_original_color = y.color
            x = y.right #TODO: What if this is nil???
            if y.parent == z: # y==y.parent.right==z.right
                x.parent = y
            else: # y == y.parent.left
                self.__rb_transplant(y, y.right)
                y.right = z.right
                y.right.parent = y

            self.__rb_transplant(z, y)
            y.left = z.left
            y.left.parent = y
            y.color = z.color

        # update sum
        self._update_parent_sum(x.parent)
        if y_original_color == BLACK:
            self.__fix_delete(x)
        self._check_properties()

    def insert(self, val, weight=1):

        new_node = ColorWeightedNode(val, weight=weight, color=RED, left=self.nil, right=self.nil)

        par = None
        curr = self.root
        while curr != self.nil:
            curr.sum += weight
            par = curr
            if val < curr.val:
                curr = curr.left
            elif val > curr.val:
                curr = curr.right
            else:
                curr.weight += weight
                return

        new_node.parent = par
        if par is None:
            self.root = new_node
        elif val < par.val:
            par.left = new_node
        else:
            par.right = new_node


        self.fix_insert(new_node)
        self._check_properties()

    def rotate_left(self, x):
        #    x    ->    y
        #     y        x
        y = x.right

        new_x_sum = x.left.sum + y.left.sum + x.weight
        new_y_sum = new_x_sum + y.weight + y.right.sum

        x.right = y.left
        if y.left != self.nil:
            y.left.parent = x

        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y
        x.sum = new_x_sum
        y.sum = new_y_sum

    def rotate_right(self, x):
        #     x    ->   y
        #    y           x
        y = x.left

        new_x_sum = x.right.sum + y.right.sum + x.weight
        new_y_sum = new_x_sum + y.weight + y.left.sum

        x.left = y.right
        if y.right != self.nil:
            y.right.parent = x

        y.parent = x.parent
        if x.parent == None:
            self.root = y
        elif x == x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y
        x.sum = new_x_sum
        y.sum = new_y_sum


    def _recolor(self, parent):
        # parent is black, need to recolor
        parent.left.color = BLACK
        parent.right.color = BLACK
        parent.color = RED
        return parent

    def fix_insert(self, curr):
        while self.root != curr and curr.parent.color == RED:
            if curr.parent == curr.parent.parent.right:
                u = curr.parent.parent.left # uncle
                if u.color == RED: # recolor
                    curr = self._recolor(u.parent)
                else: # rotate
                    if curr == curr.parent.left:
                        curr = curr.parent
                        self.rotate_right(curr)
                    curr.parent.color = BLACK
                    curr.parent.parent.color = RED
                    self.rotate_left(curr.parent.parent)
                    # The subtree's root is black so we won't need to continue
            else:
                u = curr.parent.parent.right
                if u.color == RED:
                    curr = self._recolor(u.parent)
                else:
                    if curr == curr.parent.right:
                        curr = curr.parent
                        self.rotate_left(curr)
                    curr.parent.color = BLACK
                    curr.parent.parent.color = RED
                    self.rotate_right(curr.parent.parent)
        self.root.color = BLACK



if __name__ == '__main__':
    pass