// ringzero demo (M8): `!run qjs /demo/hello.js`
const fib = (n) => (n < 2 ? n : fib(n - 1) + fib(n - 2));
const t = Date.now();
console.log(`fib(22) = ${fib(22)} in ${Date.now() - t} ms`);
const people = [
  { name: "Anna", age: 34 },
  { name: "Luca", age: 52 },
  { name: "Sara", age: 38 },
];
const avg = people.reduce((s, p) => s + p.age, 0) / people.length;
console.log(JSON.stringify({ count: people.length, avg, names: people.map((p) => p.name) }));
