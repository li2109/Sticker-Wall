var vm = new Vue({
  el: "#app",
  data: {
    colorList: [
      {
        name: "yellow",
        color: "#FFEB67"
      },
      {
        name: "blue",
        color: "#A5D8D6"
      },
      {
        name: "red",
        color: "#EF898C"
      },
      {
        name: "green",
        color: "#CBE196"
      }
    ],
    postits: [
      {
        text: "Some Text",
        color: "yellow",
        pos: { x: 20, y: 0 }
      }
    ],
    nowId: -1,
    mousePos: {
      x: 0,
      y: 0
    },
    startMousePos: {
      x: 0,
      y: 0
    }
  },
  watch: {
    postits: {
      handler: function(after, before) {
        if (this.nowId != -1) {
          console.log("push data " + this.nowId);
          postitsRef.child(this.nowId).set(this.postits[this.nowId]);
        }
      },
      deep: true
    },
    mousePos(oldPos, newPos) {
      // console.log(this.nowId)
      if (this.nowId != -1) {
        let nowPostit = this.postits[this.nowId];
        // console.log(newPos)
        nowPostit.pos.x = newPos.x - this.startMousePos.x;
        nowPostit.pos.y = newPos.y - this.startMousePos.y;
      }
    }
  },
  methods: {
    editText(pid) {
      let text = prompt("Input：", this.postits[pid].text);

      if (text != null) {
        postitsRef
          .child(pid)
          .child("text")
          .set(text);
      }
    },
    newLineToBr(text) {
      return text.replace(new RegExp("\n"), "<br>");
    },
    updateText(pid) {
      console.log("push text " + pid);
      postitsRef.child(pid).set(this.postits[pid]);
    },
    deletePostit(pid) {
      postitsRef.child(pid).remove();
    },
    postitCss(postit) {
      let startColor = this.getColor(postit.color);
      console.log(postit.color);
      let endColor = tinycolor(startColor)
        .darken(10)
        .toString();
      // console.log(`linear-gradient(${startColor} 0%,${endColor} 100%)`)
      let fontSize =
        (240 - 10) /
          (1 + Math.max(postit.text.split("\n").map(t => t.length))) -
        10;
      console.log(fontSize);
      return {
        left: postit.pos.x + "px",
        top: postit.pos.y + "px",
        "font-size": fontSize + "px",
        "line-height": fontSize + 30 + "px",
        "background-color": startColor
      };
    },
    selectId(evt, id) {
      console.log(evt);
      console.log(evt.srcElement.className);
      this.startMousePos = { x: evt.offsetX, y: evt.offsetY };
      if (
        evt.srcElement.classList.contains("block") ||
        evt.srcElement.classList.contains("fa")
      ) {
        this.nowId = -1;
      } else {
        console.log("Set id:", id);
        this.nowId = id;
      }
    },
    addPostit() {
      postitsRef.push({
        text: "Text",
        color: "yellow",
        pos: { x: 200 + Math.random() * 100, y: 200 + Math.random() * 100 }
      });
    },
    getColor(colorName) {
      let item = this.colorList.find(o => o.name == colorName);
      if (item) return item.color;
    },
    setPostitColor(pid, colorName) {
      let nowPostit = this.postits[pid];
      // console.log(newPos)
      nowPostit.color = colorName;
      console.log("push color " + pid);
      postitsRef.child(pid).set(this.postits[pid]);
    }
  }
});

window.onmousemove = evt => {
  nowMousePos = [evt].map(o => ({ x: o.pageX, y: o.pageY }))[0];
  vm.mousePos = nowMousePos;
};
window.ontouchmove = evt => {
  nowMousePos = [evt].map(o => ({ x: o.pageX, y: o.pageY }))[0];
  vm.mousePos = nowMousePos;
};
window.onmouseup = evt => {
  vm.nowId = -1;
};
window.ontouchend = evt => {
  vm.nowId = -1;
};

// Initialize Firebase
var config = {
  apiKey: "AIzaSyCidSeKYeJlTZ3eJhKntSEIvMhliKmcf5Y",
  authDomain: "sticker-wall-6631a.firebaseapp.com",
  databaseURL: "https://sticker-wall-6631a.firebaseio.com",
  projectId: "sticker-wall-6631a",
  storageBucket: "sticker-wall-6631a.appspot.com",
  messagingSenderId: "798682547239"
};
firebase.initializeApp(config);

var postitsRef = firebase.database().ref("postits");
postitsRef.on("value", function(snapshot) {
  vm.postits = snapshot.val();
});