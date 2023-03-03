(this.webpackJsonpcleanfolio=this.webpackJsonpcleanfolio||[]).push([[0],{29:function(e,t,n){},33:function(e,t,n){},34:function(e,t,n){},36:function(e,t,n){},37:function(e,t,n){},38:function(e,t,n){},39:function(e,t,n){},40:function(e,t,n){},41:function(e,t,n){},42:function(e,t,n){},43:function(e,t,n){},44:function(e,t,n){},46:function(e,t,n){"use strict";n.r(t);var c=n(9),a=n(3),i=n(1),s=n(0),r=Object(i.createContext)(),l=function(e){var t=e.children,n=localStorage.getItem("themeName"),c=Object(i.useState)(n||"dark"),l=Object(a.a)(c,2),o=l[0],j=l[1];Object(i.useEffect)((function(){}),[]);return Object(s.jsx)(r.Provider,{value:[{themeName:o,toggleTheme:function(){var e="dark"===o?"light":"dark";localStorage.setItem("themeName",e),j(e)}}],children:t})},o="/",j="KN.",u="Kirill Nagaitsev",h="Ph.D. Student at ",d="I'm a first year Ph.D. student at Northwestern University, advised by Peter Dinda. My research interests mainly lie in operating system and compiler support for parallel computing. In my free time, I enjoy running and multiplayer game development.",b="#projects",m={linkedin:"https://www.linkedin.com/in/knagaitsev/",github:"https://github.com/knagaitsev"},O=[{name:"Interrupt Polling",description:"Interrupts are a source of nondeterminism on modern architectures. This project aims to replace all hardware interrupts in a kernel with compiler-based interrupt polling at frequent intervals, removing all nondeterminism and reducing interrupt overhead.",stack:["Operating Systems","Compilers","Parallelism"]},{name:"Vlang",description:"This project aims to map high-level parallel code to hardware accelerators that can exploit the parallelism. A high-level parallel language compiles to an intermediary language that exposes the parallelism, then we map this to hardware via LLVM.",stack:["Compilers","Parallelism","Heterogeneous Hardware"]}],p=[{name:"funcX: Federated Function as a Service for Science",authors:["Zhuozhao Li","Ryan Chard","Yadu Babuji","Ben Galewsky","Tyler Skluzacek","Kirill Nagaitsev","Anna Woodard","Ben Blaiszik","Josh Bryan","Daniel S. Katz","Ian Foster","Kyle Chard"],sourceCode:"https://github.com/funcx-faas/funcX",livePreview:"https://arxiv.org/abs/2209.11631",isPublication:!0}],x=[{name:"Northwestern University",description:"Ph.D. in Computer Science",years:"2022 - Present",isPublication:!0,isEducation:!0},{name:"University of Chicago",description:"B.A. in Computer Science (magna cum laude)",years:"2018 - 2022",isPublication:!0,isEducation:!0}],f="knagaitsev@u.northwestern.edu",v=n(16),g=n.n(v),N=n(14),k=n.n(N),_=n(18),y=n.n(_),w=n(17),C=n.n(w),P=(n(29),function(){var e=Object(i.useContext)(r),t=Object(a.a)(e,1)[0],n=t.themeName,c=t.toggleTheme,l=Object(i.useState)(!1),o=Object(a.a)(l,2),j=o[0],u=o[1],h=function(){return u(!j)};return Object(s.jsxs)("nav",{className:"center nav",children:[Object(s.jsxs)("ul",{style:{display:j?"flex":null},className:"nav__list",children:[O.length?Object(s.jsx)("li",{className:"nav__list-item",children:Object(s.jsx)("a",{href:"#projects",onClick:h,className:"link link--nav",children:"Projects"})}):null,Object(s.jsx)("li",{className:"nav__list-item",children:Object(s.jsx)("a",{href:"#publications",onClick:h,className:"link link--nav",children:"Publications"})}),Object(s.jsx)("li",{className:"nav__list-item",children:Object(s.jsx)("a",{href:"#education",onClick:h,className:"link link--nav",children:"Education"})}),f?Object(s.jsx)("li",{className:"nav__list-item",children:Object(s.jsx)("a",{href:"#contact",onClick:h,className:"link link--nav",children:"Contact"})}):null]}),Object(s.jsx)("button",{type:"button",onClick:c,className:"btn btn--icon nav__theme","aria-label":"toggle theme",children:"dark"===n?Object(s.jsx)(k.a,{}):Object(s.jsx)(g.a,{})}),Object(s.jsx)("button",{type:"button",onClick:h,className:"btn btn--icon nav__hamburger","aria-label":"toggle navigation",children:j?Object(s.jsx)(C.a,{}):Object(s.jsx)(y.a,{})})]})}),S=(n(33),function(){var e=o,t=j;return Object(s.jsxs)("header",{className:"header center",children:[Object(s.jsx)("h3",{children:e?Object(s.jsx)("a",{href:e,className:"link",children:t}):t}),Object(s.jsx)(P,{})]})}),E=n(11),I=n.n(E),K=n(19),B=n.n(K),z=(n(34),n.p+"static/media/propic2.844b0895.jpeg"),D=function(){var e=u,t=h,n=d,c=b,a=m;return Object(s.jsxs)("div",{className:"about center",children:[Object(s.jsx)("div",{className:"propic-cont",children:Object(s.jsx)("img",{src:z,className:"propic propic-top",alt:"Kirill Nagaitsev"})}),e&&Object(s.jsx)("h1",{children:Object(s.jsx)("span",{className:"about__name",children:e})}),t&&Object(s.jsxs)("h2",{className:"about__role",children:[t,Object(s.jsx)("span",{className:"university",children:"Northwestern University"})]}),Object(s.jsx)("p",{className:"about__desc",children:n&&n}),Object(s.jsxs)("div",{className:"about__contact center",children:[c&&Object(s.jsx)("a",{href:c,children:Object(s.jsx)("span",{type:"button",className:"btn btn--outline",children:"Projects"})}),a&&Object(s.jsxs)(s.Fragment,{children:[a.github&&Object(s.jsx)("a",{href:a.github,"aria-label":"github",className:"link link--icon",children:Object(s.jsx)(I.a,{})}),a.linkedin&&Object(s.jsx)("a",{href:a.linkedin,"aria-label":"linkedin",className:"link link--icon",children:Object(s.jsx)(B.a,{})})]})]})]})},L=n(7),T=n.n(L),F=n(20),U=n.n(F),A=n(21),J=n.n(A),H=(n(36),function(e){var t=e.project;return Object(s.jsxs)("div",{className:"project",children:[t.isPublication?Object(s.jsx)("h4",{children:t.name}):Object(s.jsx)("h3",{children:t.name}),t.years&&Object(s.jsxs)("p",{className:"years",children:[Object(s.jsx)("span",{className:"school",children:Object(s.jsx)(U.a,{})}),Object(s.jsx)("span",{children:t.years})]}),t.isEducation&&Object(s.jsx)("hr",{}),Object(s.jsx)("p",{className:"project__description",children:t.description}),t.stack&&Object(s.jsx)("ul",{className:"project__stack",children:t.stack.map((function(e){return Object(s.jsx)("li",{className:"project__stack-item",children:e},T()())}))}),t.authors&&Object(s.jsx)("p",{children:t.authors.map((function(e,n){return Object(s.jsxs)("span",{children:["Kirill Nagaitsev"===e?Object(s.jsx)("b",{children:e}):Object(s.jsx)("span",{children:e}),n!==t.authors.length-1&&", "]},T()())}))}),t.sourceCode&&Object(s.jsx)("a",{href:t.sourceCode,"aria-label":"source code",className:"link link--icon",children:Object(s.jsx)(I.a,{})}),t.livePreview&&Object(s.jsx)("a",{href:t.livePreview,"aria-label":"live preview",className:"link link--icon",children:Object(s.jsx)(J.a,{})})]})}),M=(n(37),function(){return O.length?Object(s.jsxs)("section",{id:"projects",className:"section projects",children:[Object(s.jsx)("h2",{className:"section__title",children:"Projects"}),Object(s.jsx)("div",{className:"projects__grid",children:O.map((function(e){return Object(s.jsx)(H,{project:e},T()())}))})]}):null}),V=(n(38),function(){return p.length?Object(s.jsxs)("section",{id:"publications",className:"section publications",children:[Object(s.jsx)("h2",{className:"section__title",children:"Publications"}),Object(s.jsx)("div",{className:"publications__grid",children:p.map((function(e){return Object(s.jsx)(H,{project:e},T()())}))})]}):null}),X=(n(39),function(){return x.length?Object(s.jsxs)("section",{id:"education",className:"section education",children:[Object(s.jsx)("h2",{className:"section__title",children:"Education"}),Object(s.jsx)("div",{className:"education__grid",children:x.map((function(e){return Object(s.jsx)(H,{project:e},T()())}))})]}):null}),Y=n(22),q=n.n(Y),G=(n(40),function(){var e=Object(i.useState)(!1),t=Object(a.a)(e,2),n=t[0],c=t[1];return Object(i.useEffect)((function(){var e=function(){return window.pageYOffset>500?c(!0):c(!1)};return window.addEventListener("scroll",e),function(){return window.removeEventListener("scroll",e)}}),[]),n?Object(s.jsx)("div",{className:"scroll-top",children:Object(s.jsx)("a",{href:"#top",children:Object(s.jsx)(q.a,{fontSize:"large"})})}):null}),R=(n(41),function(){return f?Object(s.jsxs)("section",{className:"section contact center",id:"contact",children:[Object(s.jsx)("h2",{className:"section__title",children:"Contact"}),Object(s.jsx)("a",{href:"mailto:".concat(f),children:Object(s.jsx)("span",{type:"button",className:"btn btn--outline",children:"knagaitsev@u.northwestern.edu"})})]}):null}),W=(n(42),function(){return Object(s.jsx)("footer",{className:"footer",children:Object(s.jsx)("a",{href:"/",className:"link footer__link",children:"\xa9 Kirill Nagaitsev 2022"})})}),Z=(n(43),function(){var e=Object(i.useContext)(r),t=Object(a.a)(e,1)[0].themeName;return Object(s.jsxs)("div",{id:"top",className:"".concat(t," app"),children:[Object(s.jsx)(S,{}),Object(s.jsxs)("main",{children:[Object(s.jsx)(D,{}),Object(s.jsx)(M,{}),Object(s.jsx)(V,{}),Object(s.jsx)(X,{}),Object(s.jsx)(R,{})]}),Object(s.jsx)(G,{}),Object(s.jsx)(W,{})]})});n(44);Object(c.render)(Object(s.jsx)(l,{children:Object(s.jsx)(Z,{})}),document.getElementById("root"))}},[[46,1,2]]]);
//# sourceMappingURL=main.7621ea67.chunk.js.map