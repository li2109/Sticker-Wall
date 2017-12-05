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
      }, {
        text: "Something",
        color: "yellow",
        pos: { x: 20, y: 400 }
      }
    ],
    nowId : -1,
    mousePos: {
      x: 0,
      y:0
    },
    startMousePos: {
     x: 0,
     y:0
   }

  },
  watch: {
    mousePos(){
      if(this.nowId != -1)
        {
          let nowPostit = this.postits[this.nowId]
          nowPostit.pos.x = this.mousePos.x-this.startMousePos.x
          nowPostit.pos.y = this.mousePos.y-this.startMousePos.y
        }
      console.log(this.mousePos)
    }  
  },
  
  methods: {
    postitCss(p) {
      return {
       left: p.pos.x+"px",
       top: p.pos.y+"px",
        'font-size': (240-10)/(p.text.length) -10 + "px",
        backgroundColor: this.colorList.find(o=>o.name==p.color).color
      }
    },
    selectId(evt, id){
      console.log(id)
      this.nowId =id
      this.startMousePos={
        x:evt.offsetX,
        y:evt.offsetY,
      }
      console.log("start",this.startMousePos)
    }
  }
})

window.onmousemove=function (evt){
  console.log(evt)
  if(vm.nowId != -1){
      console.log(evt)
      vm.mousePos ={x: evt.pageX, y: evt.pageY}
  }

}

window.onmouseup = (evt) =>{
  console.log(evt)
  vm.nowId = -1
}