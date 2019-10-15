const a = [[1,2], [3,4]];

// Создаем rank-2 тензор
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

const model = tf.sequential({
  layers: [
    tf.layers.dense({
      inputShape: [784],
      units: 32,
      activation: 'relu'
    }),
    tf.layers.dense({
      units: 10,
      activation: 'softmax'
    })
  ]
});

model.compile({
  optimizer: 'sgd',
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy']
});

// Подготавливаем данные
const data = tf.randomNormal([100, 784]);
const labels = tf.randomNormal([100, 10]);

// Тренируем модель
model.fit(data, labels, {
  epochs: 5,
  batchSize: 32
}).then(info => {
  console.log('Точность обученной модели:', info.history.acc);
})
