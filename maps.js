function dijkstra(graph, start, end){
  let queue = [], distances = {}, prev = {};
  for(let node in graph)
    distances[node] = Infinity;
  queue.push([start, 0]);
  while(queue.length){
    queue.sort((a,b) => b[1] - a[1]);
    let [node, distance] = queue.pop();
    if(node === end){
      let path = [end];
      while(prev[node]){
        node = prev[node];
        path.push(node);
      }
      return {distance, path: path.reverse()};
    }
    for(let [nextNode, delta] of graph[node]){
      if(distance + delta < distances[nextNode]){
        distances[nextNode] = distance + delta;
        prev[nextNode] = node;
        queue.push([nextNode, distance + delta])
      }
    }
  }
}
const graph = {
  'A': [['B', 2], ['D', 3]],
  'B': [['C', 1], ['E', 3]],
  'C': [['F', 5]],
  'D': [['E', 1], ['G', 2]],
  'E': [['F', 2], ['H', 1]],
  'F': [['I', 3]],
  'G': [['H', 4]],
  'H': [['I', 1]],
  'I': []
};
const result = dijkstra(graph, 'A', 'I');
console.log(result.path.join(' -> ') + " = " + result.distance)