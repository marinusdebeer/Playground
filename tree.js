class Node {
  constructor(val){
    this.value = val;
    this.left = null;
    this.right = null;
  }
}

const root = new Node(10);

// Level 1
root.left = new Node(5);
root.right = new Node(15);

// Level 2
root.left.left = new Node(2);
root.left.right = new Node(7);
root.right.left = new Node(12);
root.right.right = new Node(18);

// Level 3
root.left.left.left = new Node(1);
root.right.left.right = new Node(14);

function traversalIterative(node){
  let queue = [node];
  while(queue.length){
    let temp = queue.pop();
    preorder.push(temp.value);
    temp.right && queue.push(temp.right);
    temp.left && queue.push(temp.left);
  }
}
let preorder = [], inorder = [], postorder = [];
console.log(traversalIterative(root));
function traversalRecursive(node){
  if(!node) return;
  preorder.push(node.value);
  node.left && traversalRecursive(node.left);
  inorder.push(node.value);
  node.right && traversalRecursive(node.right);
  postorder.push(node.value);
}
/* let preorder = [], inorder = [], postorder = [];
traversalRecursive(root);
console.log(preorder);
console.log(inorder);
console.log(postorder); */
function bfs(node){
  let queue = [node], list = []
  while(queue.length){
    let n = queue.shift();
    list.push(n.value);
    if(n.left) queue.push(n.left)
    if(n.right) queue.push(n.right)
  }
  return list;
}
// console.log(bfs(root));
function dfs(node){
  let queue = [node], list = [];
  while(queue.length){
    let n = queue.pop();
    list.push(n.value)
    if(n.right) queue.push(n.right);
    if(n.left) queue.push(n.left);
  }
return list
}
// console.log(dfs(root))