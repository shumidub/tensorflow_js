const a = [[1,2], [3,4]];

// Создаем rank-2 тензор (матрица/массив)
const b = tf.tensor([[1,2], [3,4]]);
console.log('shape:', b.shape);
b.print()

// Вывод данных
const g = tf.tensor([[1,2], [3,4]]);
g.data().then((raw) => {
  console.log('async raw value of g:', raw);
});
console.log('raw value of g:', g.dataSync());
console.log('raw multidimensional value of g:', g.arraySync());
