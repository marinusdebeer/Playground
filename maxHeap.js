class MaxHeap{
  constructor(){
    this.heap = [];
  }
  insert(val){
    this.heap.push(val);
    let index = this.heap.length-1;
    while(index > 0){
      let parentIndex = Math.floor((index - 1) / 2);
      if(this.heap[parentIndex] >= this.heap[index]) break;
      [this.heap[parentIndex], this.heap[index]] = [this.heap[index], this.heap[parentIndex]]
      index = parentIndex;
    }
  }
  remove(){
    if(this.heap.length === 0) return null;
    const max = this.heap[0];
    const last = this.heap.pop();
    if(this.heap.length === 0) return max;
    let index = 0, swapIndex = null, l = this.size();
    this.heap[0] = last;
    while(true && l > 0){
      let left = index * 2 + 1, right = index * 2 + 2;
      swapIndex = null;
      if(left < l && this.heap[left] > this.heap[index])
        swapIndex = left;
      if(right < l && this.heap[right] > (swapIndex ? this.heap[left] : this.heap[index]))
        swapIndex = right;
      if(swapIndex === null) break;
      [this.heap[index], this.heap[swapIndex]] = [this.heap[swapIndex], this.heap[index]]
      index = swapIndex;
    }

    return max;
  }
  peek(){
    return this.heap[0];
  }
  size(){
    return this.heap.length;
  }
}

let heap = new MaxHeap();
heap.insert(6);
heap.insert(4);
heap.insert(3);
heap.insert(2);
heap.insert(21);
heap.insert(12);

console.log(heap);
console.log(heap.remove())
console.log(heap.remove())
console.log(heap.remove())
console.log(heap.remove())

console.log(heap);
