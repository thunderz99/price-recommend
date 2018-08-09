import React from 'react';
import logo from './logo.svg';
import './App.css';
import fetch from 'node-fetch'
import { FormGroup, ControlLabel, FormControl, ButtonToolbar, Button, Label } from 'react-bootstrap';

class App extends React.Component {


  init_text =
    { "train_id": 1330, "name": "iPhone 6 64gb Gold", "item_condition_id": 3, "category_name": "Electronics/Cell Phones & Accessories/Cell Phones & Smartphones", "brand_name": "Apple", "price": 305.0, "shipping": 1, "item_description": "Fully functional iPhone 6 64gb Rose Gold. Only has cosmetic scratches on back. The screen is immaculate with no lifting . - all buttons are fully functional. - comes with charging cord -charging cube -original box -headphones - ottorbox - apple stickers Only selling cause I upgraded to the iPhone 7." }

  constructor() {
    super()
    this.state = { result: [] }
  }

  componentDidMount() {
    this.input.value = JSON.stringify(this.init_text, null, 2)
  }

  handleSubmit = (e) => {

    console.log(this.input.value)
    this.setState({ result: "" })

    const payload = JSON.parse(this.input.value)

    fetch('/api/predict', {
      method: 'POST',
      body: JSON.stringify(payload)
    }).then(res => {
      res.json().then(json => {
        console.log('res:' + json)
        this.setState({ result: json })
      })
    }).catch(e => {
      console.log('Error', e);
    });


  }


  renderPredictResult = () => {

    if (this.state.result.length === 0) {
      return null
    }

    const result = this.state.result

    const predict1 =
      <h3>{"" + result[0].yen.toLocaleString() + "円 と思うよ"} <br />
        {"(" + result[0].dollar.toFixed(1) + "ドル)"}
      </h3>

    return <div>
      {predict1}
    </div>

  }

  render() {

    return (
      <section>
        <div className="App">
          <header className="App-header">
            <img src={logo} className="App-logo" alt="logo" />
            <h1 className="App-title">値段予測Demo</h1>
          </header>
        </div>
        <div style={{ margin: '10px 10px 10px 10px' }} >
          <form>
            <FormGroup
              controlId="formGroup"
            >
              <ControlLabel>どれぐらい売れるかを予測する</ControlLabel>
              <FormControl
                componentClass='textarea'
                placeholder=""
                inputRef={ref => { this.input = ref }}
                rows={12}
              />
            </FormGroup>
          </form>
          <ButtonToolbar>
            <Button
              bsStyle="primary"
              onClick={this.handleSubmit}
            >予測
            </Button>
          </ButtonToolbar>
          {this.renderPredictResult()}
        </div>
      </section >
    );
  }
}

<App />

export default App;
