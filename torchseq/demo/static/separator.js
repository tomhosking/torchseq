function ping()
{
  query = "/api/ping"
  $.ajax({
    url: query,
    cache: false
  })
    .done(function( html ) {
      if(html=='ack')
      {
        $( "#connection-indicator" ).attr('class', 'good');
      }
      else {
        $( "#connection-indicator" ).attr('class', 'bad');
      }

    })
    .fail(function(){
      $( "#connection-indicator" ).attr('class', 'bad');
    });
  setTimeout(ping, 10000);
}
function getQ()
{
  query = "/api/generate?sem_input=" +encodeURIComponent($('#sem_input').val())+ "&template="+$('#template').val()
  $( "#response-spinner" ).toggleClass('d-none');
  $( "#response" ).toggleClass('d-none');
  $.ajax({
    url: query,
    cache: false
  })
    .done(function( html ) {
      $( "#response" ).html("<p>"+html+"</p>");

      $( "#response-spinner" ).toggleClass('d-none');
      $( "#response" ).toggleClass('d-none');
    })
    .fail(function(){
      $( "#response" ).html("<p>There was an error generating a question :(</p>");

      $( "#response-spinner" ).toggleClass('d-none');
      $( "#response" ).toggleClass('d-none');
    });
}
function seed(q,t)
{
  $('#sem_input').val(q);
  $('#template').val(t);
}
function seedWithMoose()
{
  q = "What is the weight of an average moose?"
  t = "How much is a surgeon's income?"
  seed(q,t)
}