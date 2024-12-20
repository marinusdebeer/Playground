//
//  ContentView.swift
//  First Vision App
//
//  Created by Marinus De Beer on 2023-06-23.
//

import SwiftUI
import RealityKit
import RealityKitContent

struct ContentView: View {
    var body: some View {
        NavigationSplitView {
            List {
                Text("Item")
            }
            .navigationTitle("Sidebar")
        } detail: {
            VStack {
                Model3D(named: "Scene", bundle: realityKitContentBundle)
                    .padding(.bottom, 500)

                Text("Welcome to my penis gallery!")
            }
            .navigationTitle("Penis Gallery")
            .padding()
        }
    }
}
