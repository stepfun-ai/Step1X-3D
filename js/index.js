import * as THREE from 'three';

import { PLYLoader } from './js/PLYLoader.js';
import { OrbitControls } from './js/OrbitControls.js'
let div_to_scene = {
    "mesh_chair_0": {
        "geo": null,
        "color": null,
    },
    "mesh_chair_1": {
        "geo": null,
        "color": null,
    },
    "mesh_chair_2": {
        "geo": null,
        "color": null,
    },
    "mesh_chair_3": {
        "geo": null,
        "color": null,
    },
    "mesh_car_0": {
        "geo": null,
        "color": null,
    },
    "mesh_car_1": {
        "geo": null,
        "color": null,
    },
    "mesh_car_2": {
        "geo": null,
        "color": null,
    },
    "mesh_car_3": {
        "geo": null,
        "color": null,
    }
}
let div_to_render_scene = {
    "mesh_style_0": {
        "0": null,
        "1": null,
        "2": null,
        "geo": null,
    },
    "mesh_style_1": {
        "0": null,
        "1": null,
        "2": null,
        "geo": null,
    },
    "mesh_style_2": {
        "0": null,
        "1": null,
        "2": null,
        "geo": null,
    },
    "mesh_style_3": {
        "0": null,
        "1": null,
        "2": null,
        "geo": null,
    },
}
let mouse_button_down = false;
let list_of_orbit_controls = []
let style_camera = null;
let render_colors = true;
let style_id = "0"

function setup_camera(div_name){
    let container = document.getElementById(div_name);
    let width = container.parentElement.clientWidth;
    let height = container.parentElement.clientHeight;
    console.log(width, height)
    let camera = new THREE.PerspectiveCamera( 35, width / height, 0.1, 50 );
    let camera_init_position = new THREE.Vector3( -1.5, 0.35, 1.2 );
    if (div_name.includes("chair")){
        camera_init_position = camera_init_position.multiplyScalar(1.5)
    }
    else if (div_name.includes("style")) {
        camera_init_position = camera_init_position.multiplyScalar(1.25)
    }
    camera.position.set(camera_init_position.x, camera_init_position.y, camera_init_position.z);
    return camera;
}

function setup_render_divs(div_name, mesh_path){
    let camera = setup_camera(div_name)
    let orbit_control = create_render_div(camera, div_name, mesh_path)
    list_of_orbit_controls.push(orbit_control)
}

function create_render_div(camera, div_id, mesh_path) {
    let container;
    let renderer, controls;

    init();
    animate();

    function init() {

        container = document.getElementById(div_id);
        let width = container.parentElement.clientWidth;
        let height = container.parentElement.clientHeight;


        div_to_scene[div_id]["color"] = new THREE.Scene();
        div_to_scene[div_id]["geo"] = new THREE.Scene();
        div_to_scene[div_id]["color"].background = new THREE.Color( 0xffffff );
        div_to_scene[div_id]["geo"].background = new THREE.Color( 0xffffff );

        // PLY file

        const loader = new PLYLoader();
        loader.load( mesh_path, function ( geometry ) {

            geometry.computeVertexNormals();
            let material_color = new THREE.MeshBasicMaterial( { color: 0xffffff, vertexColors: THREE.VertexColors} );
            let material_geo = new THREE.MeshStandardMaterial( { color: 0x444444, flatShading: true } )


            const mesh_color = new THREE.Mesh( geometry, material_color );
            const mesh_geo = new THREE.Mesh( geometry, material_geo );

            div_to_scene[div_id]["color"].add( mesh_color );
            div_to_scene[div_id]["geo"].add( mesh_geo );

        }, (xhr) => {
            console.log((xhr.loaded / xhr.total) * 100 + '% loaded')
        }, (error) => {
            console.log(error)
        }
        );

        // lights

        div_to_scene[div_id]["geo"].add( new THREE.HemisphereLight( 0x333333, 0x222222 ) );
        addShadowedLight(div_to_scene[div_id]["geo"], 1, 1, 1, 0xffffff, 1.35 );
        addShadowedLight(div_to_scene[div_id]["geo"],  0.5, 1, - 1, 0xffffff, 1 );

        // renderer

        renderer = new THREE.WebGLRenderer( { antialias: true } );
        renderer.setPixelRatio( window.devicePixelRatio );
        renderer.setSize( width, height);
        renderer.outputEncoding = THREE.sRGBEncoding;

        renderer.shadowMap.enabled = true;

        container.appendChild( renderer.domElement );

        controls = new OrbitControls(camera, renderer.domElement)
        controls.enableDamping = false

        // resize

        window.addEventListener( 'resize', onWindowResize );

}
    function onWindowResize() {
        let width = container.clientWidth;
        let height = container.clientHeight;
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize( width, height );
    }
    function animate() {
        requestAnimationFrame( animate );
        render();
    }

    function render() {
        renderer.render( div_to_scene[div_id][render_colors ? "color" : "geo"], camera );
        controls.update();
    }

    return controls;
}

function addShadowedLight(scene, x, y, z, color, intensity ) {

    const directionalLight = new THREE.DirectionalLight( color, intensity );
    directionalLight.position.set( x, y, z );
    scene.add( directionalLight );

    directionalLight.castShadow = true;

    const d = 1;
    directionalLight.shadow.camera.left = - d;
    directionalLight.shadow.camera.right = d;
    directionalLight.shadow.camera.top = d;
    directionalLight.shadow.camera.bottom = - d;

    directionalLight.shadow.camera.near = 1;
    directionalLight.shadow.camera.far = 4;

    directionalLight.shadow.mapSize.width = 1024;
    directionalLight.shadow.mapSize.height = 1024;

    directionalLight.shadow.bias = - 0.001;

}

document.addEventListener('keydown', logKey);

function logKey(evt) {
    if (evt.keyCode === 71 && !mouse_button_down) {
        switch_geometry()
    }
    if (evt.keyCode === 82 && !mouse_button_down) {
        reset_orbit_controls()
    }
}

function switch_geometry() {
    render_colors = !render_colors
}

function reset_orbit_controls() {
    list_of_orbit_controls.forEach(oc => {
        oc.reset()
    })
}

function set_style_0(){
    style_id = "0"
}

function set_style_1(){
    style_id = "1"
}

function set_style_2(){
    style_id = "2"
}

document.body.onmousedown = function(evt) {
    if (evt.button === 0)
        mouse_button_down = true
}
document.body.onmouseup = function(evt) {
    if (evt.button === 0)
        mouse_button_down = false
}

window.onload = function() {
    let slider = document.getElementsByClassName("slider")[0]
    slider.removeAttribute("tabIndex")
    // slider.addEventListener("mouseout", reset_orbit_controls);
    setup_render_divs("mesh_chair_0", './models/chair_01328.ply')
    setup_render_divs("mesh_chair_1", './models/chair_00009.ply')
    setup_render_divs("mesh_chair_2", './models/chair_00889.ply')
    setup_render_divs("mesh_chair_3", './models/chair_01829.ply')
    setup_render_divs("mesh_car_0", './models/car_00265.ply')
    setup_render_divs("mesh_car_1", './models/car_00742.ply')
    setup_render_divs("mesh_car_2", './models/car_00848.ply')
    setup_render_divs("mesh_car_3", './models/car_00944.ply')
};
