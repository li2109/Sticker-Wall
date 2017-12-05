var vm = new Vue({
  el: "#app",
  data: {
    postits: [
      {
        text: "Something",
        color: "yellow",
        pos: { x: 20, y: 0 }
      }
    ]
  },
  methods: {
    postitCss(p) {
      return {
       left: p.pos.x+"px",
       top: p.pos.y+"px",
        'font-size': (240-10)/(p.text.length) -10 + "px"
      };
    }
  }
});
