class MinHeap {
  constructor() {
    this.heap = [];
  }
  insert(value) {
    this.heap.push(value);
    this.bubbleUp();
  }
  bubbleUp() {
    let index = this.heap.length - 1;
    while (index > 0) {
      let parentIndex = Math.floor((index - 1) / 2);
      if (this.heap[parentIndex] <= this.heap[index]) break;
      [this.heap[parentIndex], this.heap[index], index] = [this.heap[index], this.heap[parentIndex], parentIndex];
    }
  }
  remove() {
    const max = this.heap[0];
    const end = this.heap.pop();
    if (this.heap.length > 0) {
      this.heap[0] = end;
      this.sinkDown();
    }
    return max;
  }
  sinkDown() {
    let index = 0;
    const length = this.heap.length;
    while (true) {
      let leftIndex = 2 * index + 1, rightIndex = 2 * index + 2, swapIndex = null;
      if (leftIndex < length && this.heap[leftIndex] < this.heap[index])
        swapIndex = leftIndex;
      if (rightIndex < length && this.heap[rightIndex] < (swapIndex === null ? this.heap[index] : this.heap[leftIndex]))
        swapIndex = rightIndex;
      if (swapIndex === null) break;
      [this.heap[index], this.heap[swapIndex], index] = [this.heap[swapIndex], this.heap[index], swapIndex];
    }
  }
  peek() {
    return this.heap[0];
  }
  size() {
    return this.heap.length;
  }
}

// Usage
const heap = new MinHeap();
heap.insert(3);
heap.insert(1);
heap.insert(4);
heap.insert(14);
heap.insert(24);
heap.insert(41);
heap.insert(9);
console.log(heap)
console.log(heap.remove());
console.log(heap.remove());
console.log(heap.remove());
console.log(heap.remove());
console.log(heap.remove());
console.log(heap.remove());
console.log(heap.remove());
