var vm = new Vue({
  el: "#app",
  data: {
    colorList:[
      {
        name: "yellow",
        color: "#FFEB67"
      },{
        name: "blue",
        color: "#A5D8D6"
      },{
        name: "red",
        color: "#EF898C"
      },{
        name: "green",
        color: "#CBE196"
      }
    ],
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
        'font-size': (240-10)/(p.text.length) -10 + "px",
        backgroundColor: this.colorList.find(o=>o.name==p.color).color
      };
    }
  }
});
