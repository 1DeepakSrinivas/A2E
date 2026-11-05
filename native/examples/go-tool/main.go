// main.go
// Minimal external tool built with the Go SDK
// Uppercase conversion and JSON Lines dispatch over stdin and stdout
package main

import (
    "bufio"
    "encoding/json"
    "fmt"
    "os"
    "strings"
)

type request struct {
    Protocol int `json:"protocol"`
    ID uint64 `json:"id"`
    Method string `json:"method"`
    Name string `json:"name"`
    Arguments json.RawMessage `json:"arguments"`
}

func main() {
    scanner := bufio.NewScanner(os.Stdin)
    scanner.Buffer(make([]byte, 4096), 262144)
    encoder := json.NewEncoder(os.Stdout)
    for scanner.Scan() {
        var r request
        if err := json.Unmarshal(scanner.Bytes(), &r); err != nil {
            fmt.Fprintln(os.Stderr, err)
            os.Exit(1)
        }
        output := any(map[string]any{})
        var failure any
        switch {
        case r.Protocol != 1:
            failure = "unsupported protocol"
        case r.Method == "init", r.Method == "snapshot":
        case r.Method == "call" && r.Name == "uppercase":
            var args struct { Text string `json:"text"` }
            if err := json.Unmarshal(r.Arguments, &args); err != nil {
                failure = "invalid arguments"
            } else {
                output = map[string]string{"text": strings.ToUpper(args.Text)}
            }
        default:
            failure = "unknown method or tool"
        }
        if err := encoder.Encode(map[string]any{"protocol": 1, "id": r.ID, "ok": failure == nil, "output": output, "error": failure}); err != nil {
            fmt.Fprintln(os.Stderr, err)
            os.Exit(1)
        }
    }
    if err := scanner.Err(); err != nil {
        fmt.Fprintln(os.Stderr, err)
        os.Exit(1)
    }
}
