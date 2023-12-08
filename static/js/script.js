let list = document.querySelectorAll('.navigation li');
let profile = document.querySelector('.profile');
let menu = document.querySelector('.menu');
profile.onclick = function () {
  menu.classList.toggle('active');
}
function activeLink() {
    list.forEach((item) =>
        item.classList.remove('active'));
    this.classList.add('active')
}
list.forEach((item) =>
    item.addEventListener('click', activeLink));

//time

let hours = document.getElementById('hour');
let minute = document.getElementById('minutes');
let seconds = document.getElementById('seconds');
let ampm = document.getElementById('ampm');

let hr = document.querySelector("#hr");
let mn = document.querySelector("#mn");
let sc = document.querySelector("#sc");


setInterval(() => {
    let h = new Date().getHours();
    let m = new Date().getMinutes();
    let s = new Date().getSeconds();

    var am = h >= 12 ? 'PM' : 'AM';

    // add zero before single digit number
    h = (h < 10) ? "0" + h : h
    m = (m < 10) ? "0" + m : m
    s = (s < 10) ? "0" + s : s


    hours.innerHTML = h
    minute.innerHTML = m;
    seconds.innerHTML = s;
    ampm.innerHTML = am;
    hr.style.transform = `rotateZ(${h * 30}deg)`;
    mn.style.transform = `rotateZ(${m * 6}deg)`;
    sc.style.transform = `rotateZ(${s * 6}deg)`;
})

document.addEventListener('DOMContentLoaded', () => {
    let text = document.getElementById('text');
    text.innerHTML = text.innerText.split('').map((char, i) =>
        `<span style="transition-delay:${i * 40}ms; filter:hue-rotate(${i * 30}deg)">${char}</span>`).join('');

    setInterval(() => {
        text.classList.toggle('h2-glow');
    }, 1700);
});

window.addEventListener("load", () => {
    const bar = document.querySelectorAll(".bar");
    for (let i = 0; i < bar.length; i++) {
        bar.forEach((item, j) => {
            // Random move
            item.style.animationDuration = `${Math.random() * (0.7 - 0.2) + 0.2}s`; // Change the numbers for speed / ( max - min ) + min / ex. ( 0.5 - 0.1 ) + 0.1
        });
    }
});

document.addEventListener('DOMContentLoaded', () => {
    const bars = document.querySelectorAll('.form-sound .bar');
    const colors = ['#ff2972', '#fee800', '#04fc43']; // Tus colores
    let cantidadDivForms = (bars.length/3) | 0;
    bars.forEach((bar, index) => {
        // bar.style.backgroundColor = colors[index % colors.length];
        if(index < cantidadDivForms){
            bar.style.backgroundColor = colors[0];
        }else if(index < cantidadDivForms*2){         
            bar.style.backgroundColor = colors[1];
        }else{
            bar.style.backgroundColor = colors[2];
        }
    });
});
