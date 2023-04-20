//
//  ContentView.swift
//  gpt_watch Watch App
//
//  Created by Marinus De Beer on 2023-04-18.
//

import SwiftUI

struct Item: Identifiable {
    let id = UUID()
    let name: String
    let details: String
}

struct ContentView: View {
    let items = [
        Item(name: "Item 1", details: "Details about Item 1."),
        Item(name: "Item 2", details: "Details about Item 2."),
        Item(name: "Item 3", details: "Details about Item 3.")
    ]
    @State var selectedItem: Item?
    
    var body: some View {
        NavigationView {
            List(items) { item in
                Button(action: {
                    selectedItem = item
                }) {
                    Text(item.name)
                }
            }
            .navigationTitle("Items")
            .sheet(item: $selectedItem) { item in
                ItemDetailsView(item: item)
            }
        }
    }
}

struct ItemDetailsView: View {
    let item: Item
    
    var body: some View {
        VStack {
            Text(item.name)
                .font(.largeTitle)
            Text(item.details)
                .padding()
            Spacer()
        }
        .navigationTitle(item.name)
    }
}


struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
